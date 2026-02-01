"""
Search the vector DB with a new image and return top-K font matches.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import open_clip
except ImportError as exc:
    raise ImportError(
        "open-clip-torch is required. Install it with: pip install open-clip-torch"
    ) from exc

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss-cpu is required. Install it with: pip install faiss-cpu"
    ) from exc


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_query_embedding(image_path, model_name, pretrained, device,
                          checkpoint_path=None, normalize=True):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )

    # Load fine-tuned weights so query embeddings match the FAISS index
    if checkpoint_path:
        cp = Path(checkpoint_path)
        if cp.exists():
            checkpoint = torch.load(cp, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded fine-tuned checkpoint: {cp}")
        else:
            print(f"WARNING: checkpoint not found at {cp}, using base weights")

    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")


def main():
    parser = argparse.ArgumentParser(description="Search font vector DB with an image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to query image (JPG/PNG)")
    parser.add_argument("--vector_db", type=str, default="vector_db",
                        help="Path to vector_db directory")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top matches to return")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cuda or cpu")
    args = parser.parse_args()

    vector_db_dir = Path(args.vector_db)
    metadata_path = vector_db_dir / "metadata.json"
    index_path = vector_db_dir / "faiss.index"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"faiss.index not found: {index_path}")

    metadata = load_metadata(metadata_path)
    model_name = metadata.get("model", "ViT-B-32")
    pretrained = metadata.get("pretrained", "openai")
    normalize = metadata.get("normalize", True)
    checkpoint = metadata.get("checkpoint")
    samples = metadata.get("samples", [])

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    query_embedding = build_query_embedding(
        args.image, model_name, pretrained, device,
        checkpoint_path=checkpoint, normalize=normalize
    )

    index = faiss.read_index(str(index_path))
    # Fetch more candidates than top_k so we can aggregate by font name
    fetch_k = min(args.top_k * 5, index.ntotal)
    scores, indices = index.search(query_embedding, fetch_k)

    # Aggregate scores by font name (best score per font)
    font_best = {}  # font_name -> (best_score, sample)
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(samples):
            continue
        sample = samples[idx]
        name = sample["font_name"]
        if name not in font_best or score > font_best[name][0]:
            font_best[name] = (float(score), sample)

    ranked = sorted(font_best.values(), key=lambda x: x[0], reverse=True)

    print(f"\nTop {args.top_k} font matches:")
    for rank, (score, sample) in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{rank}. {sample['font_name']} "
            f"(score={score:.4f}) "
            f"[sample: {sample['path']}]"
        )


if __name__ == "__main__":
    main()
