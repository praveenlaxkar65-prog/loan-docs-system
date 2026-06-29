"""
services/telegram_service.py — Store/retrieve files via Telegram Bot API.
Config (bot token + channel ID) read from DB SystemSetting table.
Managed via Admin Panel → Settings. Never hardcoded.
"""
import io
import httpx
from database import SessionLocal, get_setting


def _get_telegram_config() -> tuple[str, str]:
    """Return (bot_token, channel_id) from DB. Raises clear error if not set."""
    db = SessionLocal()
    try:
        token      = get_setting(db, "telegram_bot_token", "").strip()
        channel_id = get_setting(db, "telegram_channel_id", "").strip()
    finally:
        db.close()

    if not token:
        raise RuntimeError(
            "Telegram bot token not configured. "
            "Go to Admin Panel → Settings → Telegram and set the bot token."
        )
    if not channel_id:
        raise RuntimeError(
            "Telegram channel ID not configured. "
            "Go to Admin Panel → Settings → Telegram and set the channel ID."
        )
    return token, channel_id


async def upload_file(file_bytes: bytes, filename: str, caption: str = "") -> str:
    """
    Upload file to Telegram channel.
    Returns telegram file_id (permanent reference for later download).
    Raises RuntimeError with a clear message on config or API failure.
    """
    token, channel_id = _get_telegram_config()
    base_url = f"https://api.telegram.org/bot{token}"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    async with httpx.AsyncClient(timeout=60.0) as client:
        if ext in ("jpg", "jpeg", "png", "webp"):
            resp = await client.post(
                f"{base_url}/sendPhoto",
                data={"chat_id": channel_id, "caption": caption[:1024]},
                files={"photo": (filename, io.BytesIO(file_bytes), "image/jpeg")},
            )
        else:
            resp = await client.post(
                f"{base_url}/sendDocument",
                data={"chat_id": channel_id, "caption": caption[:1024]},
                files={"document": (filename, io.BytesIO(file_bytes), "application/octet-stream")},
            )

    if resp.status_code != 200:
        body = ""
        try:
            body = resp.json().get("description", resp.text[:200])
        except Exception:
            body = resp.text[:200]
        raise RuntimeError(
            f"Telegram API error {resp.status_code}: {body}. "
            "Check bot token and channel ID in Admin Panel → Settings → Telegram."
        )

    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(
            f"Telegram upload failed: {data.get('description', 'unknown error')}. "
            "Make sure the bot is an admin in the channel."
        )

    result = data["result"]
    if "photo" in result:
        return result["photo"][-1]["file_id"]
    elif "document" in result:
        return result["document"]["file_id"]
    raise RuntimeError("Unexpected Telegram response — no photo or document in result.")


async def download_file(file_id: str) -> bytes:
    """Download file from Telegram by file_id. Returns raw bytes."""
    token, _ = _get_telegram_config()
    base_url = f"https://api.telegram.org/bot{token}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(f"{base_url}/getFile", params={"file_id": file_id})
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram getFile failed: {data.get('description')}")
        file_path = data["result"]["file_path"]
        dl = await client.get(f"https://api.telegram.org/file/bot{token}/{file_path}")
        dl.raise_for_status()
        return dl.content


async def test_connection() -> dict:
    """
    Test Telegram config. Returns {ok, bot_username, channel_id, error}.
    """
    try:
        token, channel_id = _get_telegram_config()
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}

    base_url = f"https://api.telegram.org/bot{token}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{base_url}/getMe")
            d = r.json()
            if not d.get("ok"):
                return {"ok": False, "error": f"getMe failed: {d.get('description')}. Check bot token."}
            bot_username = d["result"].get("username", "unknown")

            # Try sending a test message to confirm channel access
            r2 = await client.post(
                f"{base_url}/sendMessage",
                json={"chat_id": channel_id, "text": "✅ LoanDocs connection test OK"},
            )
            d2 = r2.json()
            if not d2.get("ok"):
                return {
                    "ok": False,
                    "bot_username": bot_username,
                    "error": (
                        f"Bot @{bot_username} exists but cannot send to channel {channel_id}: "
                        f"{d2.get('description')}. "
                        "Make sure the bot is added as admin to the channel."
                    ),
                }
        return {"ok": True, "bot_username": bot_username, "channel_id": channel_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}
