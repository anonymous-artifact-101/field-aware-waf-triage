"""Log parsers: ModSecurity audit, Apache/NGINX combined (Kaggle web access logs), CSIC, nginx access.

Parsers read from ``data/raw/`` (immutable) and write deterministic JSONL to
``data/processed/``. Output must reproduce bit-for-bit on re-run.
"""
