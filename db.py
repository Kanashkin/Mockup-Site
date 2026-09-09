"""Database models and session setup.

Uses Postgres in production (Railway sets DATABASE_URL automatically when
its Postgres plugin is attached) and falls back to a local SQLite file for
development, so the app runs without any DB configured at all.
"""
import os
import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, LargeBinary, text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./mockup_site.db")
# Railway (and most providers) hand out "postgres://", but SQLAlchemy 1.4+
# requires the "postgresql://" scheme.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    # Null for accounts created via Google sign-in (no local password set).
    password_hash = Column(String, nullable=True)
    # Set for accounts created (or linked) via "Continue with Google".
    google_id = Column(String, unique=True, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # Free-tier usage counter (see FREE_RENDER_LIMIT in server.py) — how many
    # /render calls this account has made without an active subscription.
    # Only ever read/incremented for non-subscribed accounts; irrelevant
    # (and left as-is) once a subscription is active.
    render_count = Column(Integer, default=0, nullable=False)

    subscription = relationship("Subscription", back_populates="user", uselist=False)


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    # pending: created but not yet approved by the payer on PayPal's side.
    # active: approved and currently paid up.
    # cancelled: user (or PayPal) cancelled it.
    # expired: PayPal reports it lapsed (e.g. repeated payment failure).
    status = Column(String, default="none")
    paypal_subscription_id = Column(String, unique=True, nullable=True)
    plan_id = Column(String, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="subscription")

    def is_active(self):
        return self.status == "active"


class AnimationJob(Base):
    """One paid "Oживить" (photo -> short AI video) request. Tracks a single
    request from checkout through the async Kling (via PiAPI) generation.

    Lifecycle: pending_payment -> paid -> processing -> done|failed.
    image_data holds the rendered PNG just long enough to be served back to
    PiAPI at a public URL (see /api/animate/image/<id> in server.py) — it's
    cleared once the job reaches done/failed so finished jobs don't bloat
    the database with image bytes nobody needs anymore.
    """
    __tablename__ = "animation_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="pending_payment", nullable=False)
    # Random token (not the row id) used in the public image URL so a paid
    # job's source image isn't trivially enumerable by guessing small ints.
    image_token = Column(String, unique=True, index=True, nullable=False)
    image_data = Column(LargeBinary, nullable=True)
    image_content_type = Column(String, nullable=True)
    paypal_order_id = Column(String, unique=True, nullable=True)
    paypal_capture_id = Column(String, nullable=True)
    external_task_id = Column(String, nullable=True)
    video_url = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User")


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_google_login()
    _migrate_render_count()


def _migrate_google_login():
    """Lightweight, idempotent migration for an existing 'users' table that
    predates Google sign-in: adds the google_id column and relaxes
    password_hash to nullable. No-op on a fresh database (create_all above
    already created the table with the current, correct schema)."""
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    columns = {c["name"]: c for c in inspector.get_columns("users")}
    with engine.begin() as conn:
        if "google_id" not in columns:
            conn.execute(text("ALTER TABLE users ADD COLUMN google_id VARCHAR"))
            try:
                conn.execute(text("CREATE UNIQUE INDEX ix_users_google_id ON users (google_id)"))
            except Exception:
                pass
        if columns.get("password_hash", {}).get("nullable") is False:
            if engine.dialect.name == "postgresql":
                conn.execute(text("ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL"))
            elif engine.dialect.name == "sqlite":
                # SQLite can't drop a NOT NULL constraint in place; since this
                # only matters for local dev (production is Postgres), it's
                # simplest to leave it — local dev DBs are disposable.
                pass


def _migrate_render_count():
    """Adds the render_count column (free-tier usage counter) to an existing
    'users' table that predates it. No-op on a fresh database."""
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("users")}
    if "render_count" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN render_count INTEGER NOT NULL DEFAULT 0"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
