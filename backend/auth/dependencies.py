"""
dependencies.py — FastAPI injectable auth dependencies.

Responsibilities:
  - Extract and validate JWT bearer tokens from incoming request headers.
  - Provide get_current_user dependency for protecting any private route.
  - Provide require_admin dependency for admin-only endpoints.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from auth.jwt import decode_access_token
from core.config import Settings, get_settings

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    FastAPI dependency: extract and verify the current authenticated user.

    Reads the bearer token from the Authorization header, decodes it,
    and returns the token payload dict.

    Args:
        credentials: HTTP bearer credentials parsed by FastAPI.
        settings:    Application settings (provides JWT_SECRET).

    Returns:
        Decoded JWT payload dict with at minimum "sub" and "role" claims.

    Raises:
        HTTPException 401: If token is missing, invalid, or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(
            token=credentials.credentials,
            secret_key=settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
        )
        return payload

    except JWTError:
        raise credentials_exception


async def require_admin(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    FastAPI dependency: ensure the current user has admin privileges.

    Chains on top of get_current_user — token must be valid AND
    carry role="admin" in its payload.

    Args:
        current_user: Payload returned by get_current_user.

    Returns:
        The same current_user dict if the role check passes.

    Raises:
        HTTPException 403: If the user does not have the admin role.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user