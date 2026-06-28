"""
api/user_panel.py — Simple operator/user page for creating applications and uploading docs
"""
import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/user", tags=["User Panel"])
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "admin", "templates")


@router.get("/", response_class=HTMLResponse)
async def user_panel():
    with open(os.path.join(TEMPLATE_DIR, "user.html"), "r", encoding="utf-8") as f:
        return f.read()
