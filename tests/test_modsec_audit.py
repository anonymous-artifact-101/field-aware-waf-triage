
from __future__ import annotations

from src.data.parsers.modsec_audit import (
    MODEL_FIELDS,
    RECORD_KEYS,
    extract_bracket_kvs,
    parse_file,
    parse_transaction,
)
from src.utils.io import write_jsonl, read_jsonl

FULL_TXN = (
    "--0f0a6c3b-A--\n"
    "[01/Aug/2025:00:05:39 +0200] aIvosnASjoOLlF1qDBVj@wAAAAE 100.124.54.149 32820 100.115.127.60 443\n"
    "--0f0a6c3b-B--\n"
    "GET /a/b?x=1&y=2 HTTP/1.1\n"
    "Host: example.com\n"
    "User-Agent: Mozilla/5.0 (iPhone)\n"
    "Accept: text/html\n"
    "\n"
    "--0f0a6c3b-F--\n"
    "HTTP/1.1 301 Moved Permanently\n"
    "Content-Length: 0\n"
    "\n"
    "--0f0a6c3b-H--\n"
    'Message: Warning. Matched phrase. [id "933150"] [msg "PHP Injection"] '
    '[data "Matched: x"] [severity "CRITICAL"] [ver "OWASP_CRS/3.2.3"] '
    '[tag "attack-injection-php"] [tag "OWASP_TOP_10/A1"]\n'
    'Apache-Error: [file "x.c"] [line 273] [id "933150"] ModSecurity: Warning.\n'
    "Stopwatch: 1753999403762618 594156 (- - -)\n"
    "Stopwatch2: 1753999403762618 594156; combined=1492, p1=432, p2=913\n"
    "--0f0a6c3b-Z--\n"
)

def test_record_has_exactly_the_schema_keys():
    rec = parse_transaction(FULL_TXN, "01-Aug-2025")
    assert rec is not None
    assert set(rec.keys()) == set(RECORD_KEYS)

    assert MODEL_FIELDS == ("method", "path", "query", "ua", "status", "timing")
    for field in MODEL_FIELDS:
        assert field in rec

def test_full_transaction_values():
    rec = parse_transaction(FULL_TXN, "01-Aug-2025")
    assert rec["unique_id"] == "aIvosnASjoOLlF1qDBVj@wAAAAE"
    assert rec["timestamp"] == "2025-08-01T00:05:39+02:00"
    assert rec["day"] == "01-Aug-2025"
    assert rec["method"] == "GET"
    assert rec["path"] == "/a/b"
    assert rec["query"] == "x=1&y=2"
    assert rec["ua"] == "Mozilla/5.0 (iPhone)"
    assert rec["status"] == 301
    assert rec["timing"] == 1492
    assert rec["host"] == "example.com"
    assert rec["client_ip"] == "100.124.54.149"
    assert rec["crs_rule_ids"] == ["933150"]
    assert rec["crs_tags"] == ["attack-injection-php", "OWASP_TOP_10/A1"]
    assert rec["severities"] == ["CRITICAL"]
    assert rec["n_rules"] == 1
    assert rec["label"] == "attack"

    assert isinstance(rec["status"], int)
    assert isinstance(rec["timing"], int)

def test_apache_error_ids_do_not_duplicate_message_ids():
    rec = parse_transaction(FULL_TXN, "01-Aug-2025")

    assert rec["crs_rule_ids"].count("933150") == 1

def test_missing_user_agent_yields_empty_string():
    txn = (
        "--aaaa-A--\n"
        "[02/Aug/2025:01:00:00 +0200] uid 10.0.0.1 1 10.0.0.2 80\n"
        "--aaaa-B--\n"
        "GET /noua HTTP/1.1\n"
        "Host: h.example\n"
        "\n"
        "--aaaa-F--\n"
        "HTTP/1.1 200 OK\n"
        "--aaaa-Z--\n"
    )
    rec = parse_transaction(txn, "02-Aug-2025")
    assert rec["ua"] == ""
    assert rec["host"] == "h.example"

def test_missing_section_f_gives_status_zero():
    txn = (
        "--bbbb-A--\n"
        "[02/Aug/2025:01:00:00 +0200] uid 10.0.0.1 1 10.0.0.2 80\n"
        "--bbbb-B--\n"
        "GET /nostatus HTTP/1.1\n"
        "Host: h\n"
        "--bbbb-Z--\n"
    )
    rec = parse_transaction(txn, "02-Aug-2025")
    assert rec["status"] == 0

def test_missing_section_h_has_zero_rules_and_default_subtype():

    txn = (
        "--cccc-A--\n"
        "[02/Aug/2025:01:00:00 +0200] uid 10.0.0.1 1 10.0.0.2 80\n"
        "--cccc-B--\n"
        "GET /benign HTTP/1.1\n"
        "Host: h\n"
        "\n"
        "--cccc-F--\n"
        "HTTP/1.1 200 OK\n"
        "--cccc-Z--\n"
    )
    rec = parse_transaction(txn, "02-Aug-2025")
    assert rec["n_rules"] == 0
    assert rec["crs_rule_ids"] == []
    assert rec["label"] == "attack"
    assert rec["attack_subtype"] == "protocol"
    assert rec["timing"] == 0

def test_query_with_spaces_is_not_truncated():

    txn = (
        "--dddd-A--\n"
        "[02/Aug/2025:01:00:00 +0200] uid 10.0.0.1 1 10.0.0.2 80\n"
        "--dddd-B--\n"
        "GET /search?q=' OR 1=1-- HTTP/1.1\n"
        "Host: h\n"
        "--dddd-Z--\n"
    )
    rec = parse_transaction(txn, "02-Aug-2025")
    assert rec["method"] == "GET"
    assert rec["path"] == "/search"
    assert rec["query"] == "q=' OR 1=1--"

def test_no_query_gives_empty_query():
    txn = (
        "--eeee-A--\n"
        "[02/Aug/2025:01:00:00 +0200] uid 10.0.0.1 1 10.0.0.2 80\n"
        "--eeee-B--\n"
        "POST /upload HTTP/2\n"
        "Host: h\n"
        "--eeee-Z--\n"
    )
    rec = parse_transaction(txn, "02-Aug-2025")
    assert rec["method"] == "POST"
    assert rec["path"] == "/upload"
    assert rec["query"] == ""

def test_stopwatch_fallback_when_no_combined():
    txn = (
        "--ffff-A--\n"
        "[02/Aug/2025:01:00:00 +0200] uid 10.0.0.1 1 10.0.0.2 80\n"
        "--ffff-B--\n"
        "GET / HTTP/1.1\n"
        "Host: h\n"
        "\n"
        "--ffff-H--\n"
        'Message: x [id "1"] [severity "NOTICE"]\n'
        "Stopwatch: 1753999403762618 777 (- - -)\n"
        "--ffff-Z--\n"
    )
    rec = parse_transaction(txn, "02-Aug-2025")

    assert rec["timing"] == 777

def test_extract_bracket_kvs_handles_escaped_quotes():
    line = (
        'Message: Pattern match "a\\"b" '
        '[id "942100"] [data "Matched: \\" OR \\\\ end"] [severity "CRITICAL"]'
    )
    kvs = dict(extract_bracket_kvs(line))
    assert kvs["id"] == "942100"
    assert kvs["severity"] == "CRITICAL"

    assert '"' in kvs["data"]
    assert "\\" in kvs["data"]

def test_extract_bracket_kvs_unquoted_value():
    kvs = extract_bracket_kvs('[file "x.c"] [line 273] [level 3]')
    d = dict(kvs)
    assert d["file"] == "x.c"
    assert d["line"] == "273"
    assert d["level"] == "3"

def test_multiple_message_lines_dedupe_preserve_order():
    txn = (
        "--1111-A--\n"
        "[02/Aug/2025:01:00:00 +0200] uid 10.0.0.1 1 10.0.0.2 80\n"
        "--1111-B--\n"
        "GET / HTTP/1.1\n"
        "Host: h\n"
        "\n"
        "--1111-H--\n"
        'Message: a [id "100"] [tag "t1"] [severity "CRITICAL"]\n'
        'Message: b [id "200"] [tag "t1"] [tag "t2"] [severity "WARNING"]\n'
        'Message: c [id "100"] [tag "t3"]\n'
        "Stopwatch2: 1 2; combined=5\n"
        "--1111-Z--\n"
    )
    rec = parse_transaction(txn, "02-Aug-2025")
    assert rec["crs_rule_ids"] == ["100", "200"]
    assert rec["crs_tags"] == ["t1", "t2", "t3"]
    assert rec["severities"] == ["CRITICAL", "WARNING"]
    assert rec["n_rules"] == 2

def test_parse_file_streams_multiple_transactions(tmp_path):
    content = FULL_TXN + "\n" + FULL_TXN.replace("0f0a6c3b", "abcdef12")
    p = tmp_path / "modsec_audit.anon.log"
    p.write_text(content, encoding="utf-8", newline="")
    records = list(parse_file(p, "01-Aug-2025"))
    assert len(records) == 2
    assert records[0]["unique_id"] == "aIvosnASjoOLlF1qDBVj@wAAAAE"

def test_new_a_boundary_flushes_previous_transaction(tmp_path):

    content = (
        "--aaa1-A--\n"
        "[01/Aug/2025:00:00:01 +0200] u1 10.0.0.1 1 10.0.0.2 80\n"
        "--aaa1-B--\n"
        "GET /first HTTP/1.1\n"
        "Host: h\n"
        "--aaa2-A--\n"
        "[01/Aug/2025:00:00:02 +0200] u2 10.0.0.1 1 10.0.0.2 80\n"
        "--aaa2-B--\n"
        "GET /second HTTP/1.1\n"
        "Host: h\n"
        "--aaa2-Z--\n"
    )
    p = tmp_path / "x.log"
    p.write_text(content, encoding="utf-8", newline="")
    records = list(parse_file(p, "01-Aug-2025"))
    assert [r["path"] for r in records] == ["/first", "/second"]

def test_roundtrip_jsonl_is_deterministic(tmp_path):
    rec = parse_transaction(FULL_TXN, "01-Aug-2025")
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    write_jsonl(p1, [rec])
    write_jsonl(p2, [parse_transaction(FULL_TXN, "01-Aug-2025")])
    assert p1.read_bytes() == p2.read_bytes()
    loaded = list(read_jsonl(p1))
    assert loaded == [rec]

def test_iso_timestamp_negative_offset():
    txn = (
        "--2222-A--\n"
        "[15/Dec/2025:23:59:59 -0500] uid 10.0.0.1 1 10.0.0.2 80\n"
        "--2222-B--\n"
        "GET / HTTP/1.1\n"
        "Host: h\n"
        "--2222-Z--\n"
    )
    rec = parse_transaction(txn, "15-Dec-2025")
    assert rec["timestamp"] == "2025-12-15T23:59:59-05:00"
