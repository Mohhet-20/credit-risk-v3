"""User aggregate repository."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select

from database.models import User
from repositories.base import BaseRepository

# Columns an admin-facing "sort by" query is allowed to use -- never
# interpolate a caller-supplied string directly into `order_by`.
SORTABLE_COLUMNS = {
    "created_at": User.created_at,
    "username": User.username,
    "email": User.email,
    "role": User.role,
    "last_login_at": User.last_login_at,
}


class UserRepository(BaseRepository[User]):
    model = User

    def get_by_username(self, username: str) -> User | None:
        return self.db.scalar(select(User).where(User.username == username))

    def create(self, *, username: str, email: str, hashed_password: str, role: str = "viewer") -> User:
        user = User(username=username, email=email, hashed_password=hashed_password, role=role)
        return self.add(user)

    def list_users(self, limit: int = 100, offset: int = 0) -> list[User]:
        stmt = select(User).order_by(User.created_at.desc()).limit(limit).offset(offset)
        return list(self.db.scalars(stmt))

    def list_users_page(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        sort_by: str | None = None,
        sort_order: str = "desc",
    ) -> tuple[list[User], int]:
        """Paginated listing for the admin user-management endpoint. Returns
        `(items, total_count)` so the caller can build a `Page[T]` envelope
        without a second round trip re-deriving the count logic."""
        column = SORTABLE_COLUMNS.get(sort_by or "created_at", User.created_at)
        order = column.asc() if sort_order == "asc" else column.desc()
        stmt = select(User).order_by(order).limit(limit).offset(offset)
        items = list(self.db.scalars(stmt))
        total = self.db.scalar(select(func.count()).select_from(User)) or 0
        return items, total

    def set_active_status(self, user: User, *, is_active: bool) -> User:
        user.is_active = is_active
        self.db.commit()
        self.db.refresh(user)
        return user

    def set_role(self, user: User, *, role: str) -> User:
        user.role = role
        self.db.commit()
        self.db.refresh(user)
        return user

    def record_failed_login(self, user: User, *, max_attempts: int, lockout_minutes: int) -> User:
        """Increments the failed-attempt counter and locks the account once
        `max_attempts` is reached. Returns the (possibly now-locked) user so
        the caller can decide what error message to surface."""
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= max_attempts:
            from datetime import timedelta

            user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=lockout_minutes)
        self.db.commit()
        self.db.refresh(user)
        return user

    def record_successful_login(self, user: User) -> User:
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(user)
        return user

    def unlock_account(self, user: User) -> User:
        user.failed_login_attempts = 0
        user.locked_until = None
        self.db.commit()
        self.db.refresh(user)
        return user

    def increment_token_version(self, user: User) -> User:
        """Rotates the refresh-token version -- any refresh token minted
        before this call (including the one just used) becomes rejected by
        `AuthService.refresh`'s version check. See `database/models.py::User.token_version`."""
        user.token_version += 1
        self.db.commit()
        self.db.refresh(user)
        return user

    def set_mfa_secret(self, user: User, *, secret: str | None) -> User:
        """Stores a (not-yet-confirmed) TOTP secret. Does NOT set
        `mfa_enabled` -- see `AuthService.confirm_mfa` for the step that does."""
        user.mfa_secret = secret
        self.db.commit()
        self.db.refresh(user)
        return user

    def enable_mfa(self, user: User, *, hashed_backup_codes_json: str) -> User:
        user.mfa_enabled = True
        user.mfa_backup_codes = hashed_backup_codes_json
        self.db.commit()
        self.db.refresh(user)
        return user

    def disable_mfa(self, user: User) -> User:
        user.mfa_enabled = False
        user.mfa_secret = None
        user.mfa_backup_codes = None
        self.db.commit()
        self.db.refresh(user)
        return user

    def set_backup_codes(self, user: User, *, hashed_backup_codes_json: str | None) -> User:
        """Persists the backup-code list after one has been consumed (each
        code is single-use -- see `AuthService.verify_mfa_login`)."""
        user.mfa_backup_codes = hashed_backup_codes_json
        self.db.commit()
        self.db.refresh(user)
        return user
