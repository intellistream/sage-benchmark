"""Lightweight CPU embedding server (standalone sidecar).

Exposes an OpenAI-compatible /v1/embeddings endpoint backed by a real
sentence-transformers model running on CPU.

NOTE: This script is kept for **standalone / local development** use.
In CI (benchmark-ci.yml), the embedding engine is started automatically
by ``sagellm serve --with-embedding`` — no manual sidecar needed.

Usage (standalone):
    python scripts/embedding_server.py --port 8890 [--model <hf-model-id>]

The default model is ``sentence-transformers/all-MiniLM-L6-v2`` (~90 MB,
pure-CPU, freely available on HuggingFace).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("embedding_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="Embedding Server", version="0.1.0")

# Globally loaded model (populated on startup)
_encoder = None
_model_name: str = DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    model: str = DEFAULT_MODEL
    input: list[str] | str
    encoding_format: str = "float"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _load_model(model_name: str) -> None:
    """Load sentence-transformers model at startup."""
    global _encoder, _model_name
    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name}")
        _encoder = SentenceTransformer(model_name, device="cpu")
        _model_name = model_name
        logger.info(f"Embedding model loaded: {model_name}")
    except ImportError:
        logger.error("sentence-transformers not installed; run: pip install sentence-transformers")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Failed to load embedding model {model_name}: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "model": _model_name}


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest) -> JSONResponse:
    """OpenAI-compatible embeddings endpoint."""
    if _encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not initialised")

    texts = req.input if isinstance(req.input, list) else [req.input]
    if not texts:
        raise HTTPException(status_code=400, detail="Empty input")

    try:
        import numpy as np

        vectors: list[list[float]] = _encoder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        ).tolist()
    except Exception as exc:
        logger.error(f"Encoding failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    token_count = sum(len(t.split()) for t in texts)

    payload = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vec, "index": i} for i, vec in enumerate(vectors)
        ],
        "model": _model_name,
        "usage": {
            "prompt_tokens": token_count,
            "total_tokens": token_count,
        },
    }
    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU embedding server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("EMBEDDING_SERVER_PORT", "8890")),
        help="Port to listen on (default: 8890)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
        help=f"Sentence-transformers model to load (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _load_model(args.model)
    logger.info(f"Starting embedding server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
