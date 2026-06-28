"""
services/telegram_service.py — Store/retrieve files via Telegram Bot API
"""
import io
import httpx
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


async def upload_file(file_bytes: bytes, filename: str, caption: str = "") -> str:
    """
    Upload a file to Telegram channel.
    Returns telegram file_id (permanent reference).
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    async with httpx.AsyncClient(timeout=60.0) as client:
        if ext in ("jpg", "jpeg", "png", "webp"):
            # Send as photo for images
            resp = await client.post(
                f"{BASE_URL}/sendPhoto",
                data={"chat_id": TELEGRAM_CHANNEL_ID, "caption": caption},
                files={"photo": (filename, io.BytesIO(file_bytes), "image/jpeg")},
            )
        else:
            # Send as document for PDFs and others
            resp = await client.post(
                f"{BASE_URL}/sendDocument",
                data={"chat_id": TELEGRAM_CHANNEL_ID, "caption": caption},
                files={"document": (filename, io.BytesIO(file_bytes), "application/octet-stream")},
            )

        resp.raise_for_status()
        data = resp.json()

        if not data.get("ok"):
            raise Exception(f"Telegram upload failed: {data.get('description')}")

        result = data["result"]
        # Extract file_id from response
        if "photo" in result:
            # Photos return array — take highest resolution
            file_id = result["photo"][-1]["file_id"]
        elif "document" in result:
            file_id = result["document"]["file_id"]
        else:
            raise Exception("Unexpected Telegram response structure")

        return file_id


async def download_file(file_id: str) -> bytes:
    """
    Download file from Telegram using file_id.
    Returns raw bytes.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Get file path
        resp = await client.get(f"{BASE_URL}/getFile", params={"file_id": file_id})
        resp.raise_for_status()
        data = resp.json()

        if not data.get("ok"):
            raise Exception(f"Telegram getFile failed: {data.get('description')}")

        file_path = data["result"]["file_path"]

        # Step 2: Download actual file
        download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        file_resp = await client.get(download_url)
        file_resp.raise_for_status()

        return file_resp.content


async def test_connection() -> bool:
    """Test if Telegram bot and channel are properly configured."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BASE_URL}/getMe")
            return resp.json().get("ok", False)
    except Exception:
        return False
