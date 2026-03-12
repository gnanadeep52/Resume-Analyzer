import os
import json
import logging
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client

    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"
    if not use_vertex:
        raise EnvironmentError("GOOGLE_GENAI_USE_VERTEXAI must be TRUE")

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project:
        raise EnvironmentError("GOOGLE_CLOUD_PROJECT is not set")

    _client = genai.Client(vertexai=True, project=project, location=location)
    return _client


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) > 1 else t
        t = t.strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def generate_json(
    client: genai.Client,
    model: str,
    system: str,
    user: str,
    response_schema: dict,
    temperature: float = 0.2,
    max_retries: int = 2,
) -> dict[str, Any]:
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=response_schema,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=config,
            )
            raw = _strip_code_fences(resp.text or "")
            return json.loads(raw)
        except Exception as e:
            last_error = e
            logger.warning(f"generate_json failed attempt {attempt + 1}: {type(e).__name__}: {e}")

    return {"error": f"generate_json failed: {type(last_error).__name__}: {last_error} "}