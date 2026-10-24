"""
Out-of-band bootstrap CLI for creating the platform's first privileged
account (or any subsequent one, run by whoever has server/deploy access).

Design decision: this is a SERVER-SIDE CLI, never an HTTP endpoint, and
there is deliberately no "first user automatically becomes admin" magic
anywhere in `AuthService.register` -- a public HTTP endpoint that grants
admin to whoever's registration request happens to land first is a known
anti-pattern (a race an attacker could try to win against a fresh
deployment). Requiring `python -m scripts.create_admin` means creating a
privileged account requires the same access level as running a database
migration or reading `.env` -- i.e., already-trusted server access -- which
is the appropriate bar for minting a super_admin.

Usage:
    python -m scripts.create_admin --username admin --email admin@example.com \\
        --role super_admin [--password ...]

If `--password` is omitted, you'll be prompted (not echoed to the
terminal, via `getpass`) -- passed-on-the-command-line passwords leak into
shell history and process listings (`ps aux`), which `--password` is
provided only as a convenience for scripted/CI bootstrap where that
tradeoff is accepted deliberately.
"""

from __future__ import annotations

import argparse
import getpass
import sys

import core.sklearn_compat  # noqa: F401  (side-effect: patches XGBClassifier sklearn tags, harmless if unused here)
from api.schemas import _validate_password_strength
from core.constants import UserRole
from core.security import hash_password
from database.session import SessionLocal, init_db
from repositories.audit_log_repository import AuditLogRepository
from repositories.user_repository import UserRepository


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a privileged user account directly in the database.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--role", default=UserRole.SUPER_ADMIN.value, choices=[r.value for r in UserRole])
    parser.add_argument("--password", default=None, help="If omitted, you'll be prompted (not echoed).")
    args = parser.parse_args()

    password = args.password or getpass.getpass("Password: ")
    try:
        _validate_password_strength(password)
    except ValueError as exc:
        print(f"Password does not meet policy: {exc}", file=sys.stderr)
        return 1

    init_db()
    db = SessionLocal()
    try:
        user_repo = UserRepository(db)
        audit_repo = AuditLogRepository(db)

        if user_repo.get_by_username(args.username):
            print(f"Username '{args.username}' already exists.", file=sys.stderr)
            return 1

        user = user_repo.create(
            username=args.username,
            email=args.email,
            hashed_password=hash_password(password),
            role=args.role,
        )
        audit_repo.log_action(
            user_id=user.id,
            action="user_created_via_cli",
            resource=f"user:{user.id}",
            details={"role": args.role},
        )
        print(f"Created user '{user.username}' ({user.id}) with role '{user.role}'.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
