"""Gemini wrapper: intent parsing, translation, free-form chat."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

log = logging.getLogger("gemini")


_INTENT_SYSTEM = (
    "你是一組智慧眼鏡的語意路由器。"
    "讀取使用者的中文句子，回傳 JSON (不加 markdown)："
    '{"intent": "NAV_TO|FIND_OBJECT|TRANSLATE_TO|CANCEL|CHAT", '
    '"object": "...", "destination": "...", "lang": "..."}。'
    "intent 必填；其餘欄位只在需要時出現。"
)


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty")
        import google.generativeai as genai
        self._genai = genai
        self._genai.configure(api_key=api_key)
        self.model_name = model
        self._model = self._genai.GenerativeModel(model)

    async def parse_intent(self, text: str) -> dict[str, Any]:
        prompt = f"{_INTENT_SYSTEM}\n\n使用者: {text}\nJSON:"
        resp = await asyncio.to_thread(self._model.generate_content, prompt)
        raw = (resp.text or "").strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        try:
            return json.loads(raw)
        except Exception:
            log.warning("gemini intent parse failed: %r", raw)
            return {"intent": "CHAT"}

    async def translate(self, text: str, target_lang: str = "zh-TW") -> str:
        prompt = (f"Translate the following to {target_lang}. "
                  f"Reply with the translation only, no quotes, no commentary.\n\n{text}")
        resp = await asyncio.to_thread(self._model.generate_content, prompt)
        return (resp.text or "").strip()

    async def chat(self, text: str) -> str:
        resp = await asyncio.to_thread(self._model.generate_content, text)
        return (resp.text or "").strip()

    async def describe_scene(self, jpeg_bytes: bytes,
                             question: str = "用一句話描述前方場景。") -> str:
        """Ask Gemini about a frame — useful for fall-check, landmark lookup."""
        resp = await asyncio.to_thread(
            self._model.generate_content,
            [{"mime_type": "image/jpeg", "data": jpeg_bytes}, question],
        )
        return (resp.text or "").strip()
