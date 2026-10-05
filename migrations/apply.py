#!/usr/bin/env python
"""Apply every migration in this directory, in filename order.

No version tracking: statements are written to be re-runnable with
`create table if not exists`, and the schema is expected to be dropped and
rebuilt freely while the shape is still settling.

    DATABASE_URL=postgresql://... python migrations/apply.py
    DATABASE_URL=postgresql://... python migrations/apply.py --reset

`--reset` drops the tables first, which is the quickest way to change a column
while nothing depends on the data yet.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Dropped by --reset, referencing tables before the ones they point at.
TABLES = ["tool_runs", "workflow_runs"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="drop the tables before applying"
    )
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", ""),
        help="defaults to $DATABASE_URL",
    )
    args = parser.parse_args()

    if not args.database_url:
        print(
            "No database URL. Pass --database-url or set DATABASE_URL.\n"
            "Get one from ./postgresql.sh url",
            file=sys.stderr,
        )
        return 2

    import psycopg

    files = sorted(HERE.glob("*.sql"))
    if not files:
        print(f"No .sql files in {HERE}", file=sys.stderr)
        return 1

    with psycopg.connect(args.database_url, autocommit=True) as connection:
        if args.reset:
            for table in TABLES:
                print(f"drop {table}")
                connection.execute(f"drop table if exists {table} cascade")

        for path in files:
            print(f"apply {path.name}")
            connection.execute(path.read_text())

        rows = connection.execute(
            "select table_name from information_schema.tables "
            "where table_schema = 'public' order by table_name"
        ).fetchall()

    print(f"\ntables: {', '.join(r[0] for r in rows) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
