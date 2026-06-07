"""
jwt.py — JWT token creation and verification using python-jose.

Responsibilities:
  - Create signed access tokens with configurable expiry.
  - Verify and decode incoming bearer tokens.
  - Raise HTTP 401 on invalid or expired tokens.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

# ---------------------------------------------------------------------------
# Password hashing context (bcrypt)
# ---------------------------------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    subject: str,
    role: str,
    secret_key: str,
    algorithm: str = "HS256",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        subject:       User identifier (email or username).
        role:          User role — "admin" or "student".
        secret_key:    HMAC signing secret from environment.
        algorithm:     JWT signing algorithm.
        expires_delta: Token lifetime; defaults to 30 minutes if None.

    Returns:
        Encoded JWT string.
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=30)
    )

    payload = {
        "sub": subject,         # subject — who this token belongs to
        "role": role,           # role claim — checked by require_admin
        "exp": expire,          # expiry — jose validates this automatically
        "iat": datetime.now(timezone.utc),  # issued at — useful for audit logs
    }

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_access_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
) -> dict:
    """
    Decode and validate a JWT access token.

    Args:
        token:      Raw bearer token string (without "Bearer " prefix).
        secret_key: HMAC signing secret from environment.
        algorithm:  JWT signing algorithm.

    Returns:
        Decoded payload dict containing "sub", "role", "exp", "iat".

    Raises:
        JWTError: If the token is invalid, expired, or tampered with.
    """
    # jose raises JWTError for expired, malformed, or bad-signature tokens
    # The caller (dependencies.py) catches this and converts to HTTP 401
    payload = jwt.decode(token, secret_key, algorithms=[algorithm])

    # Ensure the token contains the minimum required claims
    if "sub" not in payload or "role" not in payload:
        raise JWTError("Token is missing required claims.")

    return payload


def hash_password(plain_password: str) -> str:
    """
    Hash a plain-text password using bcrypt.

    Args:
        plain_password: Raw password string from user input.

    Returns:
        Bcrypt hash string safe to store in the database.
    """
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against a bcrypt hash.

    Args:
        plain_password:  Raw password from login request.
        hashed_password: Stored bcrypt hash.

    Returns:
        True if password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)