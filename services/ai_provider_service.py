# -*- coding: utf-8 -*-
"""Universal AI provider service. Admin DB controls key/model/base_url/format."""
import base64, json, re
import httpx
from database import SessionLocal, AIProvider

DEFAULT_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
}

async def recognize_and_extract(image_bytes: bytes, known_doc_keys: list[str], fields_map: dict[str, list[str]]) -> dict:
    prompt = _build_prompt(known_doc_keys, fields_map)
    text = await vision_call(prompt, image_bytes)
    try:
        parsed = json.loads(_clean_json(text))
        key = parsed.get("doc_key", "unknown")
        return {
            "doc_key": key,
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "doc_display_name": parsed.get("doc_display_name") or key.replace("_", " ").title(),
            "extracted_fields": parsed.get("extracted_fields", {}) or {},
        }
    except Exception as e:
        return {"doc_key": "unknown", "confidence": 0.0, "doc_display_name": "Unknown Document", "extracted_fields": {}, "error": f"AI JSON parse failed: {e}"}

async def vision_call(prompt: str, image_bytes: bytes) -> str:
    provider = _get_active_provider()
    fmt = (provider.request_format or "openai").lower().strip()
    if not provider.api_key:
        raise RuntimeError("AI API key missing. Add and activate a provider in Admin Panel → AI Providers.")
    if fmt == "anthropic":
        return await _call_anthropic(provider, prompt, image_bytes)
    if fmt == "gemini":
        return await _call_gemini(provider, prompt, image_bytes)
    if fmt in {"openai", "custom"}:
        return await _call_openai_compatible(provider, prompt, image_bytes)
    raise RuntimeError(f"Unsupported request_format: {provider.request_format}. Use anthropic/openai/gemini/custom.")

def _get_active_provider() -> AIProvider:
    db = SessionLocal()
    try:
        provider = db.query(AIProvider).filter(AIProvider.is_active == True, AIProvider.provider_type.in_(["vision", "both"])).order_by(AIProvider.priority.asc(), AIProvider.id.asc()).first()
        if not provider:
            raise RuntimeError("No active vision AI provider found. Add one in Admin Panel → AI Providers.")
        db.expunge(provider)
        return provider
    finally:
        db.close()

def _build_prompt(known_doc_keys, fields_map):
    doc_list = "\n".join(f"- {k}" for k in known_doc_keys)
    fields_json = json.dumps(fields_map, indent=2)
    return f"""You are an expert document verification and OCR system for Indian loan processing.

Look at this image and identify the document type, then extract fields.

POSSIBLE DOCUMENT TYPES:
{doc_list}
- unknown

FIELDS TO EXTRACT PER DOCUMENT TYPE:
{fields_json}

Rules:
- Return only valid JSON, no markdown.
- Aadhaar numbers: mask first 8 digits like XXXX-XXXX-1234.
- Bank account numbers: show last 4 only like XXXXXX1234.
- Dates: DD/MM/YYYY format.
- Amounts: include INR/rupee symbol when visible.
- If field is not found: use null.

JSON schema:
{{
  "doc_key": "<exact key>",
  "confidence": <0.0-1.0>,
  "doc_display_name": "<human readable name>",
  "extracted_fields": {{"<field>": "<value or null>"}}
}}"""

async def _call_openai_compatible(provider, prompt, image_bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = provider.base_url or DEFAULT_URLS["openai"]
    payload = {"model": provider.model_name, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], "temperature": 0, "max_tokens": 1200}
    headers = {"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, json=payload); r.raise_for_status(); data = r.json()
        return data["choices"][0]["message"]["content"]

async def _call_anthropic(provider, prompt, image_bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = provider.base_url or DEFAULT_URLS["anthropic"]
    payload = {"model": provider.model_name, "max_tokens": 1200, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}, {"type": "text", "text": prompt}]}]}
    headers = {"x-api-key": provider.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, json=payload); r.raise_for_status(); data = r.json()
        return data["content"][0]["text"]

async def _call_gemini(provider, prompt, image_bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = provider.base_url or DEFAULT_URLS["gemini"].format(model=provider.model_name)
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}key={provider.api_key}" if "key=" not in url else url
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}], "generationConfig": {"temperature": 0, "maxOutputTokens": 1200}}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, json=payload); r.raise_for_status(); data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

def _clean_json(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end+1] if start != -1 and end != -1 and end > start else text
