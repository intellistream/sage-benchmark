"""Inference helpers for benchmark experiments."""

from __future__ import annotations

from typing import Any


def create_unified_inference_client(**kwargs: Any) -> Any:
    """Create an `isagellm` unified inference client with a clear error message."""
    try:
        from isagellm import UnifiedInferenceClient
    except ImportError as exc:
        raise RuntimeError(
            "isagellm is required for benchmark LLM and embedding experiments. "
            "Install it via the SAGE core dependency set or add isagellm explicitly."
        ) from exc

    return UnifiedInferenceClient.create(**kwargs)


def response_to_text(response: Any) -> str:
    """Normalize chat responses to plain text."""
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    return str(response)


def embeddings_to_list(result: Any) -> list[list[float]]:
    """Normalize embedding responses to a list of vectors."""
    if isinstance(result, list):
        return result

    content = getattr(result, "content", None)
    if isinstance(content, list):
        return content

    raise TypeError(f"Unsupported embedding response type: {type(result)!r}")
