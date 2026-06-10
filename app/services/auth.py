import hmac
import hashlib
import os
import uuid
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = Path("data/users.db")

def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            is_guest INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> tuple[str, str]:
    salt = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        100000
    ).hex()
    return pwd_hash, salt

def verify_password(password: str, pwd_hash: str, salt: str) -> bool:
    check_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        100000
    ).hex()
    return hmac.compare_digest(check_hash, pwd_hash)

def register_user(username: str, password: str) -> bool:
    username = username.strip().lower()
    if not username or not password:
        raise ValueError("Username and password cannot be empty.")
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if username.startswith("guest_"):
        raise ValueError("Usernames cannot start with 'guest_'.")
        
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            raise ValueError("Username already exists.")
            
        pwd_hash, salt = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, pwd_hash, salt)
        )
        conn.commit()
        return True
    finally:
        conn.close()

def login_user(username: str, password: str) -> str:
    username = username.strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            raise ValueError("Invalid username or password.")
            
        if not verify_password(password, row["password_hash"], row["salt"]):
            raise ValueError("Invalid username or password.")
            
        # Create session
        token = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        cursor.execute(
            "INSERT INTO sessions (token, username, is_guest, created_at) VALUES (?, ?, 0, ?)",
            (token, username, created_at)
        )
        conn.commit()
        return token
    finally:
        conn.close()

def create_guest_session() -> tuple[str, str]:
    guest_id = os.urandom(6).hex()
    username = f"guest_{guest_id}"
    token = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO sessions (token, username, is_guest, created_at) VALUES (?, ?, 1, ?)",
            (token, username, created_at)
        )
        conn.commit()
        return token, username
    finally:
        conn.close()

def get_username_from_token(token: str) -> str | None:
    if not token:
        return None
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT username FROM sessions WHERE token = ?", (token,))
        row = cursor.fetchone()
        return row["username"] if row else None
    finally:
        conn.close()

def logout_session(token: str) -> None:
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT username, is_guest FROM sessions WHERE token = ?", (token,))
        row = cursor.fetchone()
        if row:
            username = row["username"]
            is_guest = row["is_guest"]
            cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            
            # If it was a guest session, delete all their documents and chats
            if is_guest:
                logger.info(f"Cleaning up guest data for {username}")
                try:
                    from app.services.vector_store import get_vector_store
                    store = get_vector_store()
                    store.delete_user_data(username)
                except Exception as e:
                    logger.error(f"Failed to delete guest data for {username}: {e}")
    finally:
        conn.close()

def cleanup_old_guest_sessions() -> None:
    """Find and delete guest sessions older than 4 hours."""
    cutoff_time = (datetime.utcnow() - timedelta(hours=4)).isoformat()
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT token, username FROM sessions WHERE is_guest = 1 AND created_at < ?", (cutoff_time,))
        expired_sessions = cursor.fetchall()
        
        if expired_sessions:
            from app.services.vector_store import get_vector_store
            store = get_vector_store()
            
            for session in expired_sessions:
                token = session["token"]
                username = session["username"]
                logger.info(f"Cleaning up expired guest session for {username}")
                try:
                    store.delete_user_data(username)
                except Exception as e:
                    logger.error(f"Failed to delete expired guest data for {username}: {e}")
                cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
    finally:
        conn.close()
