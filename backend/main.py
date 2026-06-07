"""
main.py — FastAPI application entry point for the KUST RAG API.

Responsibilities:
  - Instantiate the FastAPI application with metadata and CORS settings.
  - Register all routers (chat, admin, transcribe, auth).
  - Initialise shared resources at startup and verify connections.
  - Expose /health endpoint for container health checks.
"""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from pinecone import Pinecone

from auth.jwt import create_access_token, verify_password
from core.config import Settings, get_settings
from ingest.embedder import ensure_index
from routes import admin, chat, transcribe


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Runs once on startup, yields to serve requests, runs cleanup on shutdown.
    """
    settings = get_settings()

    # --- Startup ---
    print(f"[main] Starting {settings.app_name}...")

    # Verify OpenAI connectivity
    try:
        openai_client = OpenAI(api_key=settings.openai_api_key)
        openai_client.models.list()  # lightweight ping
        print("[main] OpenAI connection verified.")
    except Exception as e:
        print(f"[main] WARNING: OpenAI connection failed: {e}")

    # Verify Pinecone connectivity + ensure index exists
    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        ensure_index(pc, settings.pinecone_index)
        print(f"[main] Pinecone index '{settings.pinecone_index}' ready.")
    except Exception as e:
        print(f"[main] WARNING: Pinecone connection failed: {e}")

    print(f"[main] {settings.app_name} is ready.")

    yield  # — app is live and serving requests —

    # --- Shutdown ---
    print(f"[main] Shutting down {settings.app_name}...")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Build and configure the FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="RAG-powered AI assistant for Kohat University of Science & Technology",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",       # Swagger UI at /docs
        redoc_url="/redoc",     # ReDoc at /redoc
    )

    # CORS — tightened to specific origins via settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(chat.router)
    app.include_router(admin.router)
    app.include_router(transcribe.router)
    app.include_router(auth_router)

    return app


# ---------------------------------------------------------------------------
# Auth router — defined inline since it's small and needs jwt + settings
# ---------------------------------------------------------------------------

from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


SEED_USERS = {
    "admin@kust.edu.pk":   {"password_hash": "", "role": "admin"},
    "student@kust.edu.pk": {"password_hash": "", "role": "student"},
}


@auth_router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """
    Authenticate a user and return a signed JWT access token.

    For Phase 1 this checks against hardcoded seed users.
    Replace SEED_USERS with a PostgreSQL lookup in Phase 2.
    """
    user = SEED_USERS.get(payload.email)

    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        subject=payload.email,
        role=user["role"],
        secret_key=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )

    print(f"[auth] Login successful: {payload.email} (role={user['role']})")

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
    )


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health_check() -> dict:
    """
    Liveness probe for Docker and Render health checks.
    UptimeRobot should ping this every 5 minutes to keep the free tier warm.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": "1.0.0",
    }
