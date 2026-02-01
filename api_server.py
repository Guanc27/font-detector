"""
Font Detection API Server.

FastAPI application that accepts an image upload and returns the top-K
matching Google Fonts using an OpenCLIP embedding model and FAISS index.

Run locally:
    uvicorn api_server:app --reload --port 8000

Test:
    curl -X POST http://localhost:8000/detect -F "file=@test.png"
"""

import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import torch
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

import open_clip
from preprocessing import FontImagePreprocessor

# ---------------------------------------------------------------------------
# Configuration (environment variables with sensible local defaults)
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "models/best_model.pt")
VECTOR_DB_DIR = os.environ.get("VECTOR_DB_DIR", "vector_db")
MODEL_NAME = os.environ.get("MODEL_NAME", "ViT-B-32")
PRETRAINED = os.environ.get("PRETRAINED", "openai")
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "5"))

# ---------------------------------------------------------------------------
# Shared state populated once at startup
# ---------------------------------------------------------------------------
state: dict = {}


def _google_fonts_url(font_name: str) -> str:
    """Build a Google Fonts specimen URL from a font name."""
    slug = font_name.replace(" ", "+")
    return f"https://fonts.google.com/specimen/{slug}"


def _load_resources():
    """Load model, FAISS index, metadata, and preprocessor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- OpenCLIP model --
    model, _, _preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device,
    )

    checkpoint_path = Path(CHECKPOINT_PATH)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = model.to(device)
    model.eval()

    # -- FAISS index + metadata --
    db_dir = Path(VECTOR_DB_DIR)
    index_path = db_dir / "faiss.index"
    metadata_path = db_dir / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    samples = metadata.get("samples", [])
    normalize = metadata.get("normalize", True)

    # -- Preprocessor --
    preprocessor = FontImagePreprocessor(target_size=224)

    return {
        "model": model,
        "device": device,
        "index": index,
        "samples": samples,
        "metadata": metadata,
        "normalize": normalize,
        "preprocessor": preprocessor,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup; release on shutdown."""
    state.update(_load_resources())
    yield
    state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Font Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chrome extensions use chrome-extension:// origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Return model / index status."""
    if not state:
        return JSONResponse({"status": "loading"}, status_code=503)

    metadata = state["metadata"]
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "checkpoint": CHECKPOINT_PATH,
        "device": str(state["device"]),
        "index_vectors": state["index"].ntotal,
        "font_count": len(metadata.get("font_to_id", {})),
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=50),
    mode: str = Query("single", pattern="^(single|multi)$"),
):
    """Identify the font in an uploaded image.

    Parameters
    ----------
    file : uploaded image (PNG / JPEG / WebP / etc.)
    top_k : number of results to return (default 5)
    mode : ``"single"`` (pre-cropped region, default) or ``"multi"``
           (full photo — runs EasyOCR text detection first)
    """
    start = time.perf_counter()

    # -- Read & validate image --
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        # Re-open after verify (verify consumes the stream)
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(
            {"error": "Invalid image file."}, status_code=400,
        )

    preprocessor: FontImagePreprocessor = state["preprocessor"]
    model = state["model"]
    device = state["device"]
    index = state["index"]
    samples = state["samples"]
    normalize = state["normalize"]

    # -- Preprocess --
    if mode == "multi":
        tensor = preprocessor.preprocess_for_model(image)
    else:
        tensor = preprocessor.preprocess_single(image)

    regions_detected = tensor.shape[0]

    # -- Embed --
    tensor = tensor.to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)

    # Average embeddings when multiple regions are detected
    if features.shape[0] > 1:
        features = features.mean(dim=0, keepdim=True)

    query = features.cpu().numpy().astype("float32")

    # -- Search FAISS --
    fetch_k = min(top_k * 5, index.ntotal)
    scores, indices = index.search(query, fetch_k)

    # Aggregate best score per font
    font_best: dict[str, tuple[float, dict]] = {}
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(samples):
            continue
        sample = samples[int(idx)]
        name = sample["font_name"]
        if name not in font_best or score > font_best[name][0]:
            font_best[name] = (float(score), sample)

    ranked = sorted(font_best.values(), key=lambda x: x[0], reverse=True)

    matches = []
    for rank, (score, sample) in enumerate(ranked[:top_k], start=1):
        matches.append({
            "rank": rank,
            "font_name": sample["font_name"],
            "score": round(score, 4),
            "google_fonts_url": _google_fonts_url(sample["font_name"]),
        })

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "matches": matches,
        "processing_time_ms": round(elapsed_ms, 1),
        "regions_detected": regions_detected,
    }
