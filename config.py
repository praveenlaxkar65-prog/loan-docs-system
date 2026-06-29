"""
config.py — Minimal app config from .env.
AI provider keys and Telegram config are managed via Admin Panel (stored in DB).
Only truly static/startup config lives here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Admin panel auth ─────────────────────────────────────────────
SECRET_KEY      = os.getenv("SECRET_KEY", "changeme_secret_key_please_set_in_env")
ADMIN_USERNAME  = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD  = os.getenv("ADMIN_PASSWORD", "admin123")

# ── App metadata ─────────────────────────────────────────────────
APP_NAME        = os.getenv("APP_NAME", "LoanDocs System")
APP_HOST        = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT        = int(os.getenv("APP_PORT", 8000))
DEBUG           = os.getenv("DEBUG", "true").lower() == "true"

# ── Application ID prefix ────────────────────────────────────────
APP_ID_PREFIX   = "LOAN"

# ── Upload limits ────────────────────────────────────────────────
MAX_FILE_SIZE_MB       = 20
ALLOWED_EXTENSIONS     = {".jpg", ".jpeg", ".png", ".pdf", ".webp", ".heic"}

# ── Image processing ─────────────────────────────────────────────
ENHANCED_IMAGE_QUALITY = 95
MAX_IMAGE_DIMENSION    = 2000   # px
