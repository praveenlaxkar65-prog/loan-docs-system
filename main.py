"""
main.py — LoanDocs System entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import APP_NAME, DEBUG
from database import init_db
from api.upload import router as upload_router
from api.admin import router as admin_api_router
from api.admin_panel import router as admin_panel_router
from api.user_panel import router as user_panel_router
from api.test_panel import router as test_panel_router

# ── Init DB ──────────────────────────────────────────────────────
init_db()

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_NAME,
    description="Loan Document Management System — AI powered document recognition and extraction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────
app.include_router(upload_router)
app.include_router(admin_api_router)
app.include_router(admin_panel_router)
app.include_router(user_panel_router)
app.include_router(test_panel_router)


# ── Health ───────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "app": APP_NAME}


@app.get("/")
async def root():
    return {
        "app": APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "admin": "/admin",
        "user": "/user",
        "test": "/test",
        "health": "/health",
    }