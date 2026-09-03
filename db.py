"""Database models and session setup.

Uses Postgres in production (Railway sets DATABASE_URL automatically when
its Postgres plugin is attached) and falls back to a local SQLite file for
development, so the app runs without any DB configured at all.
"""
import os
import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, text, inspect
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


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_google_login()


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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
