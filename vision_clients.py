#!/usr/bin/env python3
"""
Vision model client abstractions for DiagramLens.

This module decouples the markdown-processing workflow from any specific
provider transport so we can swap Ollama for hosted multimodal services
without rewriting the diagram analysis logic.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OPENCODE_GO_MESSAGES_URL = "https://opencode.ai/zen/go/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def _load_image_as_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _detect_media_type(image_path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(image_path))
    if guessed:
        return guessed
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "application/octet-stream"


def _validate_image(image_path: Path) -> None:
    Image.open(image_path).verify()


class VisionClient(ABC):
    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        image_path: Optional[Path] = None,
        temperature: float = 0.0,
    ) -> str:
        pass


class OllamaVisionClient(VisionClient):
    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL):
        self.base_url = base_url

    def generate(
        self,
        model: str,
        prompt: str,
        image_path: Optional[Path] = None,
        temperature: float = 0.0,
    ) -> str:
        message = {"role": "user", "content": prompt}

        if image_path:
            _validate_image(image_path)
            message["images"] = [_load_image_as_base64(image_path)]

        payload = {
            "model": model,
            "messages": [message],
            "options": {"temperature": temperature},
            "stream": False,
        }

        resp = requests.post(self.base_url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()


class OpenCodeGoVisionClient(VisionClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENCODE_GO_MESSAGES_URL,
    ):
        self.api_key = (
            api_key
            or os.environ.get("OPENCODE_API_KEY")
            or os.environ.get("OPENCODE_GO_API_KEY")
        )
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("OpenCode Go requiere OPENCODE_API_KEY u OPENCODE_GO_API_KEY")

    def generate(
        self,
        model: str,
        prompt: str,
        image_path: Optional[Path] = None,
        temperature: float = 0.0,
    ) -> str:
        content: list[dict[str, object]] = []

        if image_path:
            _validate_image(image_path)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _detect_media_type(image_path),
                        "data": _load_image_as_base64(image_path),
                    },
                }
            )

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    self.base_url,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": ANTHROPIC_VERSION,
                        "content-type": "application/json",
                    },
                    json=payload,
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
                blocks = data.get("content", [])
                texts = [block.get("text", "") for block in blocks if block.get("type") == "text"]
                return "\n".join(item for item in texts if item).strip()
            except requests.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code is not None and status_code < 500:
                    raise
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise

        if last_error is not None:
            raise last_error
        return ""


def create_vision_client(
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> VisionClient:
    provider_key = provider.strip().casefold()
    if provider_key == "ollama":
        return OllamaVisionClient(base_url=base_url or DEFAULT_OLLAMA_URL)
    if provider_key in {"opencode-go", "opencode_go", "opencode"}:
        return OpenCodeGoVisionClient(
            api_key=api_key,
            base_url=base_url or DEFAULT_OPENCODE_GO_MESSAGES_URL,
        )
    raise ValueError(f"Proveedor visual no soportado: {provider}")
