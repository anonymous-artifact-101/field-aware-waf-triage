
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines._common import (
    git_commit, load_split_records, render_record, render_records, select_label_budget, subtype_labels,
)

def render_records_list(recs):
    return list(render_records(recs))
from src.eval.aggregate import aggregate_table, write_cell_file
from src.eval.latency import percentiles
from src.eval.efficiency import mean_ci95_runs
from src.utils.config import load_config
from src.utils.paths import RESULTS_DIR
from src.utils.runlog import append_run, dataset_version
from src.utils.seeds import set_seed

_TABLE = "table_08_latency"
_STATS = ("mean", "p50", "p95", "p99")
_MB = 1024.0 * 1024.0

def _time_records(
    recs: Sequence[Mapping[str, Any]],
    featurize: Callable[[Mapping[str, Any]], Any],
    score: Callable[[Any], Any],
    *,
    warmup: int,
    n_trials: int,
) -> Dict[str, Dict[str, float]]:
    import gc as _gc

    timed = list(recs[warmup : warmup + max(1, n_trials)])
    samples: Dict[str, List[float]] = {"featurize": [], "score": [], "end_to_end": []}

    for r in recs[:warmup]:
        score(featurize(r))
    _gc.collect()
    _gc.disable()
    try:
        for r in timed:
            t0 = time.perf_counter()
            feat = featurize(r)
            _ = score(feat)
            samples["end_to_end"].append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            feat = featurize(r)
            samples["featurize"].append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            _ = score(feat)
            samples["score"].append((time.perf_counter() - t0) * 1000.0)
    finally:
        _gc.enable()
    return {k: percentiles(v) for k, v in samples.items()}

def _build_baseline(name: str, labeled, labels, seed: int) -> Tuple[Callable, Callable, float]:
    if name in ("tfidf_logreg", "tfidf_svd_hgbdt", "isolation_forest", "ocsvm"):

        texts = render_records_list(labeled)
        if name == "tfidf_logreg":
            from src.baselines.tfidf_logreg import fit_tfidf_logreg
            vec, est = fit_tfidf_logreg(texts, labels, {}, {}, seed)
            score = lambda x: est.predict(x)
        elif name == "tfidf_svd_hgbdt":
            from src.baselines.tfidf_svd_hgbdt import fit_tfidf_svd_hgbdt
            model = fit_tfidf_svd_hgbdt(
                texts,
                labels,
                {
                    "learning_rate": 0.08,
                    "max_iter": 300,
                    "max_leaf_nodes": 31,
                    "min_samples_leaf": 20,
                    "class_weight": "balanced",
                    "early_stopping": False,
                },
                {
                    "ngram_range": [1, 2],
                    "analyzer": "char_wb",
                    "max_features": 50000,
                    "svd_components": 256,
                },
                seed,
            )
            featurizer = model[:-1]
            est = model.named_steps["hgbdt"]
            featurize = lambda rec: featurizer.transform([render_record(rec)])
            score = lambda x: est.predict(x)
            fp = len(pickle.dumps(model)) / _MB
            return featurize, score, fp
        elif name == "isolation_forest":
            from src.baselines.isolation_forest import fit_unsupervised_text_detector
            vec, est = fit_unsupervised_text_detector(texts, {}, {}, seed)
            score = lambda x: est.score_samples(x)
        else:
            from src.baselines.ocsvm import fit_ocsvm
            vec, est = fit_ocsvm(texts, {}, {})
            score = lambda x: est.decision_function(x)
        featurize = lambda rec: vec.transform([render_record(rec)])
        fp = (len(pickle.dumps(vec)) + len(pickle.dumps(est))) / _MB
        return featurize, score, fp

    if name in ("tfidf_typed_linearsvc", "tfidf_flat_linearsvc"):

        from src.baselines.tfidf_typed_linearsvc import fit_typed_tfidf_linearsvc
        _cfg_path = ("configs/baselines/tfidf_flat_linearsvc.yaml"
                     if name == "tfidf_flat_linearsvc"
                     else "configs/baselines/tfidf_typed_linearsvc.yaml")
        cfg = dict(load_config(_cfg_path))
        enc, est = fit_typed_tfidf_linearsvc(
            labeled, labels, dict(cfg.get("model", {})), dict(cfg.get("features", {})), seed)
        featurize = lambda rec: enc.transform([rec])
        score = lambda x: est.predict(x)
        fp = (len(pickle.dumps(enc)) + len(pickle.dumps(est))) / _MB
        return featurize, score, fp

    if name == "field_prefixed_fasttext":
        from src.baselines.field_prefixed_fasttext import fit_field_prefixed_fasttext

        cfg = dict(load_config("configs/baselines/field_prefixed_fasttext.yaml"))
        enc, est = fit_field_prefixed_fasttext(
            labeled, labels, dict(cfg.get("model", {})), dict(cfg.get("features", {})), seed)
        featurize = lambda rec: enc.transform([rec])
        score = lambda x: est.predict(x)
        fp = (len(pickle.dumps(enc)) + len(pickle.dumps(est))) / _MB
        return featurize, score, fp

    if name == "hashing_char_sgd":
        from src.baselines.hashing_char_sgd import HashingCharLinearPredictor, fit_hashing_char_sgd

        cfg = dict(load_config("configs/baselines/hashing_char_sgd.yaml"))
        pipe = fit_hashing_char_sgd(
            render_records_list(labeled), labels,
            dict(cfg.get("model", {})), dict(cfg.get("features", {})), seed)
        fast = HashingCharLinearPredictor(pipe)
        featurize = lambda rec: fast.featurize_text(render_record(rec))
        score = lambda x: fast.predict_from_features(x)
        fp = len(pickle.dumps(pipe)) / _MB
        return featurize, score, fp

    if name in ("modsec_learn", "modsec_advlearn"):
        from src.baselines.modsec_learn import CrsFiringVectorizer
        if name == "modsec_learn":
            from src.baselines.modsec_learn import fit_modsec_learn
            vec, est = fit_modsec_learn(labeled, labels, {}, {"rule_firing_source": "crs_rule_ids"}, seed)
        else:
            from src.baselines.modsec_advlearn import fit_modsec_advlearn
            vec, est = fit_modsec_advlearn(
                labeled, labels, {"num_classes": 8}, {"rule_firing_source": "crs_rule_ids"},
                {"enabled": True, "epsilon": 0.1, "n_steps": 7, "train_ratio": 0.5}, seed)
        featurize = lambda rec: vec.transform([rec])
        score = lambda x: est.predict(x)
        fp = (len(pickle.dumps(vec)) + len(pickle.dumps(est))) / _MB
        return featurize, score, fp

    if name == "status_only":
        from src.baselines.status_only import fit_status_rule, _status_of
        rule, majority = fit_status_rule(labeled, labels)
        featurize = lambda rec: _status_of(rec)
        score = lambda s: rule.get(s, majority)
        fp = (len(pickle.dumps(rule)) + len(pickle.dumps(majority))) / _MB
        return featurize, score, fp

    if name == "char_cnn":

        import io

        import torch

        from src.baselines.char_cnn import CharCNN, _train_cnn, encode_char_batch
        from src.data.labels import SUBTYPES

        torch.set_num_threads(1)
        cfg = dict(load_config("configs/baselines/char_cnn.yaml"))
        features = dict(cfg.get("features", {}))
        model_cfg = dict(cfg.get("model", {}))
        train_cfg = dict(cfg.get("train", {}))
        vocab_size = int(features.get("vocab_size", 128))
        max_length = int(features.get("max_length", 256))
        dev = torch.device("cpu")

        model = CharCNN(
            vocab_size=vocab_size,
            embed_dim=int(model_cfg.get("embed_dim", 64)),
            conv_channels=int(model_cfg.get("conv_channels", 128)),
            kernel_sizes=[int(k) for k in model_cfg.get("kernel_sizes", [3, 5, 7])],
            num_classes=int(model_cfg.get("num_classes", len(SUBTYPES))),
            dropout=float(model_cfg.get("dropout", 0.2)),
        ).to(dev)
        train_ids = encode_char_batch(render_records_list(labeled), vocab_size, max_length)
        train_labels = torch.tensor(labels, dtype=torch.long)
        _train_cnn(model, train_ids, train_labels, train_cfg, dev)
        model.eval()

        def featurize(rec):
            return encode_char_batch([render_record(rec)], vocab_size, max_length)

        @torch.no_grad()
        def score(ids):
            return int(model(ids.to(dev)).argmax(dim=-1).item())

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        fp = buf.getbuffer().nbytes / _MB
        return featurize, score, fp

    raise ValueError(f"unknown / non-per-record baseline {name!r}")

_ROWS = {
    "tfidf_logreg": "TF-IDF + LogReg (latency)",
    "tfidf_svd_hgbdt": "TF-IDF + SVD + HGBDT (latency)",
    "tfidf_typed_linearsvc": "TF-IDF typed + LinearSVC (latency)",
    "tfidf_flat_linearsvc": "TF-IDF flat + LinearSVC (latency)",
    "field_prefixed_fasttext": "Field-prefixed FastText + LinearSVC (latency)",
    "hashing_char_sgd": "Hashing char-ngram + SGD (latency)",
    "isolation_forest": "Isolation Forest (latency)",
    "ocsvm": "One-Class SVM (latency)",
    "modsec_learn": "ModSec-Learn (latency)",
    "modsec_advlearn": "ModSec-AdvLearn (latency)",
    "status_only": "Status-only (shortcut) (latency)",
    "char_cnn": "Char-CNN (latency)",
}
_DEFAULT = list(_ROWS)

def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Per-record CPU latency of the per-record baselines (Table 8).")
    parser.add_argument("--baselines", nargs="*", default=_DEFAULT, help="Subset to time.")
    parser.add_argument("--efficiency", default="configs/eval/efficiency.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=None,
                        help="Independent timing runs per baseline for the latency CI "
                             "(default: efficiency.yaml latency.runs or 10). HGBDT and "
                             "the tree/forest scorers carry real run-to-run variance, so "
                             "a single run understates their tail; the CI exposes it.")
    parser.add_argument("--max-records", type=int, default=None, help="Cap (smoke only).")
    parser.add_argument("--budget", type=float, default=0.10,
                        help="Label budget the baseline is fit on before timing (default 0.10, "
                             "the manuscript headline budget). The featurizer vocabulary "
                             "(and thus featurize cost) reflects that fit.")
    parser.add_argument("--no-aggregate", action="store_true")
    args = parser.parse_args(argv)

    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    eff = dict(load_config(args.efficiency)).get("efficiency", {})
    lat = dict(eff.get("latency", {}))
    warmup, n_trials = int(lat.get("warmup", 20)), int(lat.get("n_trials", 300))
    runs = int(args.runs) if args.runs is not None else int(lat.get("runs", 10))
    split = str(lat.get("split", "owasp_test"))
    seed = int(args.seed)
    set_seed(seed)

    train = load_split_records("owasp_train", max_records=args.max_records)
    test = load_split_records(split, max_records=args.max_records)
    idx = select_label_budget(len(train), float(args.budget))
    labeled = [train[i] for i in idx]
    labels = subtype_labels(labeled)

    table_dir = RESULTS_DIR / _TABLE
    commit, data_ver = git_commit(), dataset_version()
    hardware = os.environ.get("PECTI_HARDWARE", "CPU (set $PECTI_HARDWARE)")
    n_written = 0

    for name in args.baselines:
        try:
            featurize, score, fp_mb = _build_baseline(name, labeled, labels, seed)
        except Exception as exc:
            print(f"[54] skip {name}: {exc}")
            continue

        import gc as _gc
        per_run = {s: {"mean": [], "p50": [], "p95": [], "p99": []} for s in ("end_to_end", "featurize", "score")}
        last_stages = {}
        for _ in range(max(1, runs)):
            _gc.collect()
            st = _time_records(test, featurize, score, warmup=warmup, n_trials=n_trials)
            last_stages = st
            for s in per_run:
                for k in ("mean", "p50", "p95", "p99"):
                    per_run[s][k].append(st[s][k])
        agg = {s: {k: mean_ci95_runs(per_run[s][k]) for k in ("mean", "p50", "p95", "p99")} for s in per_run}
        row = _ROWS[name]
        for stat in _STATS:
            a = agg["end_to_end"][stat]
            val = a["mean"]
            ci_hw = round(float(a["ci95"]), 6) if stat != "p50" else None
            if val is None:
                continue
            meta = {
                "commit": commit, "gpu": "cpu", "dataset_version": data_ver,
                "metric": "latency_ms", "unit": "milliseconds",
                "single_run_measured": False, "n_runs": int(runs),
                "ci95_halfwidth": ci_hw,
                "eval_capped": args.max_records is not None, "baseline": name,
                "fit_budget": float(args.budget),
                "stages_featurize": {k: round(last_stages["featurize"][k], 6) for k in _STATS},
                "stages_score": {k: round(last_stages["score"][k], 6) for k in _STATS},
                "run_std_ms": round(float(agg["end_to_end"]["mean"]["std"]), 6),
                "num_threads": 1, "split": split, "hardware": hardware,
                "note": "per-record baseline latency over %d independent runs; "
                        "DeepLog/LogBERT excluded (windowed / not CPU-runnable)." % runs,
            }
            write_cell_file(table_dir, table=_TABLE, row=row, col=stat,
                            seed=seed, value=round(float(val), 6), metadata=meta)
            n_written += 1
        stages = last_stages

        write_cell_file(table_dir, table=_TABLE, row="Footprint (MB)", col=row.replace(" (latency)", ""),
                        seed=seed, value=round(fp_mb, 4),
                        metadata={"commit": commit, "gpu": "cpu", "dataset_version": data_ver,
                                  "metric": "footprint_mb", "unit": "megabytes",
                                  "single_run_measured": True, "eval_capped": False,
                                  "baseline": name, "component": name})
        n_written += 1
        e2e = stages["end_to_end"]
        print(f"[54] {row:30s} end_to_end mean={e2e['mean']:.4f}ms p95={e2e['p95']:.4f}ms fp={fp_mb:.3f}MB")

    print(f"[54] wrote {n_written} cell(s) under {table_dir / 'cells'}")
    try:
        append_run(stage="baseline_latency_table8", config_path=args.efficiency, seed=seed,
                   artifact=str(table_dir / "results.json"),
                   notes=f"baselines={','.join(args.baselines)}",
                   extra={"eval_capped": args.max_records is not None})
    except Exception as exc:
        print(f"[54] WARN ledger: {exc}")
    if not args.no_aggregate:
        aggregate_table(table_dir, commit=commit, gpu="cpu", round_to=None)
        print("[54] aggregated; render: python scripts/50_make_paper_tables.py --table 8 --allow-exploratory")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
