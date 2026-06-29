# -*- coding: utf-8 -*-
"""
services/ai_provider_service.py
────────────────────────────────
Universal AI provider service.
- Zero hardcoded keys or URLs.
- All config comes from DB (AIProvider table), managed via Admin Panel.
- Supports: anthropic / openai / gemini / custom (openai-compatible)
- AI Gateway (Google Apps Script or any proxy) = just another provider with request_format=openai/custom.
- Fallback chain: tries providers in priority order; moves to next on any failure.
- Normalized output always:
    {success, provider, model, confidence, doc_key, extracted_fields, raw_response, error}
"""

import base64
import json
import re
import httpx
from database import SessionLocal, AIProvider

# ── DEFAULT URLS (only used if base_url is blank in DB) ──────────
_DEFAULT_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai":    "https://api.openai.com/v1/chat/completions",
    "gemini":    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
}


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

async def recognize_and_extract(
    image_bytes: bytes,
    known_doc_keys: list[str],
    fields_map: dict[str, list[str]],
) -> dict:
    """
    Main entry point called by upload pipeline.
    Tries providers in priority order until one succeeds.
    Returns normalized dict always — never raises.
    """
    prompt = _build_prompt(known_doc_keys, fields_map)
    result = await vision_call_with_fallback(prompt, image_bytes)

    if not result["success"]:
        return {
            "doc_key": "unknown",
            "confidence": 0.0,
            "doc_display_name": "Unknown Document",
            "extracted_fields": {},
            "error": result.get("error", "All AI providers failed"),
            "provider": result.get("provider", "none"),
        }

    raw_text = result["raw_response"] or ""
    try:
        parsed = json.loads(_clean_json(raw_text))
        key = parsed.get("doc_key") or "unknown"
        return {
            "doc_key": key,
            "confidence": float(parsed.get("confidence") or 0.0),
            "doc_display_name": parsed.get("doc_display_name") or key.replace("_", " ").title(),
            "extracted_fields": parsed.get("extracted_fields") or {},
            "provider": result["provider"],
            "model": result["model"],
        }
    except Exception as e:
        return {
            "doc_key": "unknown",
            "confidence": 0.0,
            "doc_display_name": "Unknown Document",
            "extracted_fields": {},
            "error": f"AI response parse failed: {e}",
            "provider": result["provider"],
            "raw_response": raw_text[:500],
        }


async def vision_call_with_fallback(prompt: str, image_bytes: bytes) -> dict:
    """
    Try each active vision provider in priority order.
    Returns normalized result dict from the first that succeeds.
    If all fail, returns last error in normalized format.
    """
    providers = _get_active_providers()
    if not providers:
        return _error_result("none", "", "No active vision AI provider. Add one in Admin Panel → AI Providers.")

    last_error = ""
    for provider in providers:
        result = await _call_provider(provider, prompt, image_bytes)
        if result["success"]:
            return result
        last_error = result.get("error", "Unknown error")
        # Continue to next provider

    return _error_result(
        providers[-1].provider_name if providers else "none",
        providers[-1].model_name if providers else "",
        f"All {len(providers)} provider(s) failed. Last error: {last_error}",
    )


async def test_provider(provider_id: int) -> dict:
    """
    Test a single provider by ID with a minimal dummy image.
    Returns {success, provider, error, response_preview}.
    """
    db = SessionLocal()
    try:
        provider = db.query(AIProvider).filter(AIProvider.id == provider_id).first()
        if not provider:
            return {"success": False, "error": "Provider not found"}
        db.expunge(provider)
    finally:
        db.close()

    # 1x1 white JPEG as test image
    test_image = bytes([
        0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
        0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
        0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
        0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
        0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
        0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
        0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
        0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
        0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
        0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
        0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,0x00,0x02,0x01,0x03,
        0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,
        0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,
        0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,
        0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,
        0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,
        0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,
        0xFF,0xDA,0x00,0x08,0x01,0x01,0x00,0x00,0x3F,0x00,0xFB,0xD2,
        0x8A,0x28,0x03,0xFF,0xD9,
    ])

    test_prompt = 'This is a test. Reply ONLY with this JSON: {"status": "ok", "message": "provider works"}'
    result = await _call_provider(provider, test_prompt, test_image)
    result["response_preview"] = (result.get("raw_response") or "")[:300]
    result.pop("raw_response", None)
    return result


# ═══════════════════════════════════════════════════════════════════
# PROVIDER DISPATCH
# ═══════════════════════════════════════════════════════════════════

async def _call_provider(provider: "AIProvider", prompt: str, image_bytes: bytes) -> dict:
    """Dispatch to correct adapter based on request_format. Returns normalized dict."""
    fmt = (provider.request_format or "openai").lower().strip()

    if not provider.api_key:
        return _error_result(provider.provider_name, provider.model_name,
                             "API key missing. Set it in Admin Panel → AI Providers.")
    if not provider.model_name:
        return _error_result(provider.provider_name, provider.model_name,
                             "Model name missing. Set it in Admin Panel → AI Providers.")

    timeout = float(provider.timeout_seconds or 90)

    try:
        if fmt == "anthropic":
            raw = await _adapter_anthropic(provider, prompt, image_bytes, timeout)
        elif fmt == "gemini":
            raw = await _adapter_gemini(provider, prompt, image_bytes, timeout)
        elif fmt in ("openai", "custom"):
            raw = await _adapter_openai(provider, prompt, image_bytes, timeout)
        else:
            return _error_result(provider.provider_name, provider.model_name,
                                 f"Unsupported request_format '{fmt}'. Use: anthropic / openai / gemini / custom")

        return {
            "success": True,
            "provider": provider.provider_name,
            "model": provider.model_name,
            "raw_response": raw,
            "confidence": 0.0,
            "doc_key": "",
            "extracted_fields": {},
            "error": None,
        }

    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text[:400]
        except Exception:
            pass
        return _error_result(provider.provider_name, provider.model_name,
                             f"HTTP {e.response.status_code}: {body}")
    except httpx.TimeoutException:
        return _error_result(provider.provider_name, provider.model_name,
                             f"Request timed out after {timeout}s")
    except Exception as e:
        return _error_result(provider.provider_name, provider.model_name, str(e))


# ═══════════════════════════════════════════════════════════════════
# ADAPTERS — one per request_format
# ═══════════════════════════════════════════════════════════════════

async def _adapter_anthropic(provider, prompt: str, image_bytes: bytes, timeout: float) -> str:
    """Anthropic claude-* format."""
    b64 = base64.b64encode(image_bytes).decode()
    url = (provider.base_url or "").strip() or _DEFAULT_URLS["anthropic"]
    payload = {
        "model": provider.model_name,
        "max_tokens": 1200,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    headers = {
        "x-api-key": provider.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["content"][0]["text"]


async def _adapter_openai(provider, prompt: str, image_bytes: bytes, timeout: float) -> str:
    """
    OpenAI / OpenAI-compatible format.
    Covers: OpenAI, OpenRouter, AI Gateway (Google Apps Script), any proxy.
    The AI Gateway just needs to accept the same JSON body and return choices[0].message.content.
    """
    b64 = base64.b64encode(image_bytes).decode()
    url = (provider.base_url or "").strip() or _DEFAULT_URLS["openai"]

    payload = {
        "model": provider.model_name,
        "max_tokens": 1200,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    }

    # Auth header — provider can set custom auth in api_key field as "Bearer token" or plain token
    api_key = provider.api_key or ""
    auth_header = api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"

    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }

    # Extra headers from notes field: "X-Header: value" lines
    for line in (provider.notes or "").splitlines():
        if ":" in line and line.startswith("X-"):
            k, _, v = line.partition(":")
            headers[k.strip()] = v.strip()

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _adapter_gemini(provider, prompt: str, image_bytes: bytes, timeout: float) -> str:
    """Google Gemini native format."""
    b64 = base64.b64encode(image_bytes).decode()
    base = (provider.base_url or "").strip() or _DEFAULT_URLS["gemini"].format(model=provider.model_name)
    sep = "&" if "?" in base else "?"
    url = f"{base}{sep}key={provider.api_key}" if "key=" not in base else base

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            ],
        }],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1200},
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]


# ═══════════════════════════════════════════════════════════════════
# DB HELPERS
# ═══════════════════════════════════════════════════════════════════

def _get_active_providers() -> list["AIProvider"]:
    """Return all active vision/both providers ordered by priority asc."""
    db = SessionLocal()
    try:
        providers = (
            db.query(AIProvider)
            .filter(
                AIProvider.is_active == True,
                AIProvider.provider_type.in_(["vision", "both"]),
            )
            .order_by(AIProvider.priority.asc(), AIProvider.id.asc())
            .all()
        )
        # expunge so they're usable outside session
        for p in providers:
            db.expunge(p)
        return providers
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_prompt(known_doc_keys: list[str], fields_map: dict[str, list[str]]) -> str:
    doc_list    = "\n".join(f"- {k}" for k in known_doc_keys)
    fields_json = json.dumps(fields_map, indent=2)
    return f"""You are an expert document verification and OCR system for Indian loan processing.

Look at this image and identify the document type, then extract the relevant fields.

POSSIBLE DOCUMENT TYPES (return ONLY one of these exact keys):
{doc_list}
- unknown

FIELDS TO EXTRACT PER DOCUMENT TYPE:
{fields_json}

Rules:
- Return ONLY valid JSON — no markdown, no explanation.
- Aadhaar numbers: mask first 8 digits → XXXX-XXXX-1234
- Bank account numbers: show last 4 only → XXXXXX1234
- Dates: DD/MM/YYYY format
- Amounts: include ₹ symbol when visible
- If a field is not found or not visible: use null

Required JSON schema:
{{
  "doc_key": "<exact key from list above>",
  "confidence": <0.0 to 1.0>,
  "doc_display_name": "<human readable name>",
  "extracted_fields": {{
    "<field_name>": "<value or null>"
  }}
}}"""


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _clean_json(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _error_result(provider: str, model: str, error: str) -> dict:
    return {
        "success": False,
        "provider": provider,
        "model": model,
        "raw_response": None,
        "confidence": 0.0,
        "doc_key": "unknown",
        "extracted_fields": {},
        "error": error,
    }
