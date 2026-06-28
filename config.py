"""
config.py — Central configuration loaded from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID  = os.getenv("TELEGRAM_CHANNEL_ID", "")
SECRET_KEY           = os.getenv("SECRET_KEY", "changeme_secret_key")
ADMIN_USERNAME       = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD       = os.getenv("ADMIN_PASSWORD", "admin123")
APP_NAME             = os.getenv("APP_NAME", "LoanDocs System")
APP_HOST             = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT             = int(os.getenv("APP_PORT", 8000))
DEBUG                = os.getenv("DEBUG", "true").lower() == "true"

# App ID prefix
APP_ID_PREFIX = "LOAN"

# Upload limits
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".webp", ".heic"}

# Image processing
ENHANCED_IMAGE_QUALITY = 95
MAX_IMAGE_DIMENSION    = 2000   # px — resize if larger
