from __future__ import annotations

import argparse
import sys

from services.api_keys import create_api_key, init_db, list_api_keys, revoke_api_key


def _cmd_create(args: argparse.Namespace) -> int:
    row = create_api_key(args.name)
    print("API key created. Save it now; it will not be shown again.")
    print(f"id: {row['id']}")
    print(f"name: {row['name']}")
    print(f"prefix: {row['key_prefix']}")
    print(f"key: {row['key']}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    rows = list_api_keys(include_revoked=args.all)
    if not rows:
        print("No API keys found.")
        return 0
    for row in rows:
        status = "revoked" if row.get("revoked_at") else "active"
        print(
            f"{row['id']}\t{status}\t{row['key_prefix']}\t{row['name']}\t"
            f"created={row['created_at']}\tlast_used={row.get('last_used_at') or '-'}"
        )
    return 0


def _cmd_revoke(args: argparse.Namespace) -> int:
    if revoke_api_key(args.identifier):
        print("API key revoked.")
        return 0
    print("API key not found or already revoked.")
    return 1


def _cmd_init(_: argparse.Namespace) -> int:
    init_db()
    print("API key database is ready.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage Rozeeta API keys.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a new API key.")
    p_create.add_argument("name", help="Human-readable key name.")
    p_create.set_defaults(func=_cmd_create)

    p_list = sub.add_parser("list", help="List API keys without secret values.")
    p_list.add_argument("--all", action="store_true", help="Include revoked keys.")
    p_list.set_defaults(func=_cmd_list)

    p_revoke = sub.add_parser("revoke", help="Revoke a key by id, name, or prefix.")
    p_revoke.add_argument("identifier", help="Key id, name, or visible prefix.")
    p_revoke.set_defaults(func=_cmd_revoke)

    p_init = sub.add_parser("init", help="Create the API key database.")
    p_init.set_defaults(func=_cmd_init)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
