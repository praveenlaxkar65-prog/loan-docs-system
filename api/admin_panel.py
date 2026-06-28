"""
api/admin_panel.py — Serve admin panel HTML
"""
import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/admin", tags=["Admin Panel"])
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "admin", "templates")


@router.get("/", response_class=HTMLResponse)
async def admin_panel():
    with open(os.path.join(TEMPLATE_DIR, "index.html"), "r") as f:
        return f.read()
