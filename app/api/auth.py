from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Depends, status, Request, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings
from app.models.schemas import UserAuthRequest, AuthResponse
from app.services.auth import (
    register_user,
    login_user,
    create_guest_session,
    logout_session,
    get_username_from_token,
    list_registered_users,
)

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Dependency to validate the HTTP Bearer session token."""
    token = credentials.credentials
    username = get_username_from_token(token)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username

@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
def register(req: UserAuthRequest):
    """Register a new user account."""
    try:
        register_user(req.username, req.password)
        return {"message": "User registered successfully"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/auth/login", response_model=AuthResponse)
def login(req: UserAuthRequest):
    """Authenticate a user and start a session."""
    try:
        token = login_user(req.username, req.password)
        return AuthResponse(token=token, username=req.username.lower(), is_guest=False)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/auth/guest", response_model=AuthResponse)
def guest_login():
    """Create a temporary guest session."""
    try:
        token, username = create_guest_session()
        return AuthResponse(token=token, username=username, is_guest=True)
    except Exception as e:
        logger.error(f"Guest login error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/auth/logout")
def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Terminate the active session (and clear guest files if guest)."""
    token = credentials.credentials
    try:
        logout_session(token)
        return {"message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/admin/users")
def get_users_list(secret: str | None = Query(default=None)):
    """Secret endpoint to list registered users and their session states."""
    if not settings.admin_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is disabled because admin_secret is not configured."
        )
    if secret != settings.admin_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin secret."
        )
    try:
        return list_registered_users()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
