"""Runtime stores the API writes to.

Everything the API *serves* is precomputed by the pipeline and read-only at
request time. The two things a running system creates -- a person's own
check-in answers and the record of who looked at whose case -- need a write
path, and it must not live in ``data/processed/``, which every pipeline run
rewrites wholesale.

    access_log.py   who viewed which pseudonym's record, when, and whether
                    the request was granted (SQLite)

Check-in answers still live in ``backend/api/checkin_store.py`` as an
append-only JSONL file; that module's docstring says why.
"""
