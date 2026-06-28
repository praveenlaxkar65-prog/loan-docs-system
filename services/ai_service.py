"""
services/ai_service.py — Claude Vision for doc recognition + field extraction
"""
import base64
import json
import httpx
from config import CLAUDE_API_KEY

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-sonnet-4-6"

HEADERS = {
    "x-api-key": CLAUDE_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}


# ──────────────────────────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────

async def recognize_document(image_bytes: bytes, known_doc_keys: list[str]) -> dict:
    """
    Step 1: Identify what document this is.
    Returns { doc_key, confidence, display_name }
    """
    doc_list = "\n".join(f"- {k}" for k in known_doc_keys)

    prompt = f"""You are a document verification expert for Indian loan processing.

Look at this image carefully and identify what type of document it is.

POSSIBLE DOCUMENT TYPES (return ONLY one of these exact keys):
{doc_list}
- unknown

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "doc_key": "<exact key from list above>",
  "confidence": <0.0 to 1.0>,
  "reason": "<one line why you identified it as this>"
}}"""

    response = await _claude_vision_call(prompt, image_bytes)
    try:
        return json.loads(_clean_json(response))
    except Exception:
        return {"doc_key": "unknown", "confidence": 0.0, "reason": "Parse error"}


async def extract_fields(image_bytes: bytes, doc_key: str, fields: list[str]) -> dict:
    """
    Step 2: Extract specific fields from identified document.
    Returns { field_name: value, ... }
    """
    fields_list = "\n".join(f"- {f}" for f in fields)

    prompt = f"""You are an expert OCR system for Indian documents.

This is a {doc_key.replace('_', ' ').title()} document.

Extract the following fields as accurately as possible:
{fields_list}

Rules:
- For Aadhaar numbers: mask first 8 digits as XXXX-XXXX, show last 4 only (e.g. XXXX-XXXX-1234)
- For account numbers: show only last 4 digits masked (e.g. XXXXXX1234)
- If a field is not visible or not found, use null
- For dates: use DD/MM/YYYY format
- For amounts: include currency symbol and use numbers only (e.g. ₹45000)

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "<field_name>": "<extracted value or null>",
  ...
}}"""

    response = await _claude_vision_call(prompt, image_bytes)
    try:
        return json.loads(_clean_json(response))
    except Exception:
        return {f: None for f in fields}


async def recognize_and_extract(
    image_bytes: bytes,
    known_doc_keys: list[str],
    fields_map: dict[str, list[str]]
) -> dict:
    """
    Combined call: recognize + extract in one shot.
    fields_map = { doc_key: [field1, field2, ...] }
    Returns { doc_key, confidence, extracted_fields }
    """
    doc_list    = "\n".join(f"- {k}" for k in known_doc_keys)
    fields_json = json.dumps(fields_map, indent=2)

    prompt = f"""You are an expert document verification and OCR system for Indian loan processing.

Look at this image and:
1. Identify what document type it is
2. Extract all relevant fields

POSSIBLE DOCUMENT TYPES:
{doc_list}
- unknown

FIELDS TO EXTRACT PER DOCUMENT TYPE:
{fields_json}

Rules for extraction:
- Aadhaar numbers: mask first 8 digits → XXXX-XXXX-1234
- Bank account numbers: show last 4 only → XXXXXX1234
- Dates: DD/MM/YYYY format
- Amounts: include ₹ symbol
- If field not found: use null
- Be precise, read carefully

Respond ONLY with valid JSON:
{{
  "doc_key": "<exact key>",
  "confidence": <0.0-1.0>,
  "doc_display_name": "<human readable name>",
  "extracted_fields": {{
    "<field>": "<value or null>"
  }}
}}"""

    response = await _claude_vision_call(prompt, image_bytes)
    try:
        result = json.loads(_clean_json(response))
        return result
    except Exception:
        return {
            "doc_key": "unknown",
            "confidence": 0.0,
            "doc_display_name": "Unknown Document",
            "extracted_fields": {}
        }


# ──────────────────────────────────────────────────────────────────
# PRIVATE
# ──────────────────────────────────────────────────────────────────

async def _claude_vision_call(prompt: str, image_bytes: bytes) -> str:
    """Call Claude Vision API with an image."""
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(CLAUDE_API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]


def _clean_json(text: str) -> str:
    """Strip markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    return text
