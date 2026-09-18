from __future__ import annotations

import os
from typing import Any

import requests


class BackendClient:
    def __init__(self, base_url: str | None = None, timeout: int = 150) -> None:
        self.base_url = (base_url or os.getenv("TRUTHLENS_BACKEND_URL", "http://127.0.0.1:5000")).rstrip("/")
        self.timeout = timeout

    def verify_text(self, claim: str) -> dict[str, Any]:
        return self._post("/api/v1/verify", json={"claim": claim})

    def verify_image(self, image_bytes: bytes, filename: str = "telegram-photo.jpg") -> dict[str, Any]:
        return self._post("/api/v1/verify/image", files={"image": (filename, image_bytes, "image/jpeg")})

    def _post(self, path: str, **kwargs) -> dict[str, Any]:
        try:
            response = requests.post(f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
            data = response.json()
            if not response.ok:
                raise RuntimeError(data.get("error", "Verification failed."))
            return data
        except requests.RequestException as error:
            raise RuntimeError("TruthLens verification service is unavailable.") from error
