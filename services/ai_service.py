"""Backward-compatible shim. New code should import from ai_provider_service directly."""
from services.ai_provider_service import recognize_and_extract, vision_call_with_fallback as vision_call
