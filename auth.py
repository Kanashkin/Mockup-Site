"""Email+password auth with a signed session cookie (no JWT, no separate
session table — the cookie itself, signed by starlette's SessionMiddleware,
holds the user id). Simple and enough for a single-server deployment."""
import re

import bcrypt
from fastapi import Request, HTTPException, Depends
from sqlalchemy.orm import Session

from db import get_db, User, Subscription

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# bcrypt's underlying algorithm only uses the first 72 bytes of the input;
# anything past that is silently ignored on both hash and verify, which is
# harmless in practice (72 bytes is already a very long password) but worth
# truncating explicitly so it never raises on an unusually long paste.
def _truncate(password: str) -> bytes:
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_truncate(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        # Google-only accounts have no local password to check against.
        return False
    try:
        return bcrypt.checkpw(_truncate(password), password_hash.encode("utf-8"))
    except ValueError:
        return False


def validate_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Invalid email address")
    return email


def validate_password(password: str) -> str:
    if not password or len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    return password


def create_user_session(request: Request, user_id: int):
    request.session["user_id"] = user_id


def clear_user_session(request: Request):
    request.session.pop("user_id", None)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(401, "Not logged in")
    return user


def require_active_subscription(user: User = Depends(require_user), db: Session = Depends(get_db)) -> User:
    sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    if not sub or not sub.is_active():
        raise HTTPException(402, "Active subscription required")
    return user
