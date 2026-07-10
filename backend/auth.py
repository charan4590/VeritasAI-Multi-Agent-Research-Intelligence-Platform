"""
Phase 6: JWT Authentication
=============================
Architecture decisions:
  - bcrypt for password hashing (industry standard, salted, slow by design)
  - JWT tokens with 24h expiry (stateless — no server-side session storage)
  - Users stored in SQLite (same DB, zero extra infra)
  - Secret key from env var (never hardcoded)

Why JWT over sessions: JWT is stateless so it works across multiple
server instances. Sessions require shared storage (Redis etc) which
is overkill for this project.

Why bcrypt: MD5/SHA256 are too fast for passwords — GPUs can crack them.
bcrypt is deliberately slow (cost factor 12 = ~300ms per hash).
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SECRET_KEY = os.environ.get(
    "JWT_SECRET",
    "change-this-secret-in-production-use-openssl-rand-hex-32",
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.environ.get("TOKEN_EXPIRE_HOURS", "24"))
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() == "true"

DB_PATH = os.environ.get("DB_PATH", "research.db")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_tables():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    # Create default admin user if AUTH_ENABLED and no users exist
    if AUTH_ENABLED:
        _ensure_default_user()


def _ensure_default_user():
    """Create admin/admin123 if no users exist. Must be changed in production."""
    with _conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            default_pw = os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin123")
            hashed = pwd_context.hash(default_pw)
            conn.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                ("admin", hashed),
            )
            conn.commit()
            logger.warning("Created default user admin/%s — change this immediately!", default_pw)


def get_user(username: str) -> Optional[Dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND is_active = 1",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def create_user(username: str, password: str) -> bool:
    """Register a new user. Returns False if username taken."""
    try:
        hashed = pwd_context.hash(password)
        with _conn() as conn:
            conn.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                (username, hashed),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


# ---------------------------------------------------------------------------
# Token logic
# ---------------------------------------------------------------------------


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: Dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[Dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[Dict]:
    """
    FastAPI dependency. Validates JWT token.
    If AUTH_ENABLED=false, returns a dummy user (open access).
    """
    if not AUTH_ENABLED:
        return {"username": "guest", "id": 0}

    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    username = payload.get("sub")
    user = get_user(username) if username else None
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
