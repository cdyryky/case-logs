from __future__ import annotations

import argparse
import json

from .constants import DEFAULT_DB_PATH
from .export_payload import export_approved_json
from .importer import import_xlsx
from .models import connect, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="ACGME IR case-log automation CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db")
    import_cmd = sub.add_parser("import-xlsx")
    import_cmd.add_argument("xlsx_path")
    export_cmd = sub.add_parser("export-approved")
    export_cmd.add_argument("--output", default=None)
    args = parser.parse_args()

    conn = connect(DEFAULT_DB_PATH)
    try:
        init_db(conn)
        if args.command == "init-db":
            print(f"Initialized {DEFAULT_DB_PATH}")
        elif args.command == "import-xlsx":
            summary = import_xlsx(conn, args.xlsx_path)
            print(json.dumps(summary, indent=2))
        elif args.command == "export-approved":
            out = export_approved_json(conn, args.output)
            print(out)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()

