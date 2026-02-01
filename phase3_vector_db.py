"""
Phase 3: Vector Database Setup
Generates embeddings for all font samples and builds a FAISS index.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ImportError as exc:
    raise ImportError(
        "open-clip-torch is required for Phase 3. "
        "Install it with: pip install open-clip-torch"
    ) from exc

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss-cpu is required for Phase 3. "
        "Install it with: pip install faiss-cpu"
    ) from exc


class FontSampleDataset(Dataset):
    """Dataset that returns preprocessed images and sample metadata."""

    def __init__(self, samples, dataset_dir, preprocess):
        self.samples = samples
        self.dataset_dir = Path(dataset_dir)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.dataset_dir / sample["path"].replace("\\", "/")
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        return image, sample


def collate_batch(batch):
    images, metas = zip(*batch)
    return torch.stack(images), list(metas)


def load_samples(metadata_file):
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    samples = []
    font_to_id = {}
    for font_idx, font_data in enumerate(metadata["fonts"]):
        font_name = font_data["name"]
        font_to_id[font_name] = font_idx
        for sample in font_data["samples"]:
            sample_entry = {
                "font_name": font_name,
                "font_id": font_idx,
                "path": sample["path"],
                "text": sample.get("text", ""),
                "size": sample.get("size"),
            }
            samples.append(sample_entry)

    return samples, font_to_id


def filter_existing_samples(samples, dataset_dir):
    dataset_dir = Path(dataset_dir)
    valid = []
    missing = 0
    for sample in samples:
        image_path = dataset_dir / sample["path"].replace("\\", "/")
        if image_path.exists():
            valid.append(sample)
        else:
            missing += 1
    if missing:
        print(f"⚠️  Skipped {missing} missing images from metadata.")
    return valid


def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Build FAISS vector database")
    parser.add_argument("--dataset_dir", type=str, default="font_dataset",
                        help="Path to font dataset directory")
    parser.add_argument("--metadata", type=str, default="font_dataset/metadata.json",
                        help="Path to metadata.json")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="OpenCLIP model name")
    parser.add_argument("--pretrained", type=str, default="openai",
                        help="Pretrained weights (should match Phase 2)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding generation")
    parser.add_argument("--output_dir", type=str, default="vector_db",
                        help="Output directory for FAISS index and metadata")
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Also save embeddings to embeddings.npy")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable L2 normalization (not recommended)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model + preprocess
    print(f"Loading OpenCLIP model: {args.model}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model = model.to(device)
    model.eval()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Load samples
    samples, font_to_id = load_samples(args.metadata)
    samples = filter_existing_samples(samples, args.dataset_dir)
    if not samples:
        raise RuntimeError("No valid samples found. Check dataset paths and metadata.")

    dataset = FontSampleDataset(samples, args.dataset_dir, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_batch)

    embeddings_list = []
    metadata_list = []
    normalize = not args.no_normalize

    print(f"Generating embeddings for {len(samples)} samples...")
    with torch.no_grad():
        for images, metas in tqdm(loader, desc="Embedding"):
            images = images.to(device)
            features = model.encode_image(images)
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
            embeddings = features.cpu().numpy().astype("float32")
            embeddings_list.append(embeddings)

            # Attach a stable sample_id for lookup
            for meta in metas:
                meta_copy = dict(meta)
                meta_copy["sample_id"] = len(metadata_list)
                metadata_list.append(meta_copy)

    all_embeddings = np.vstack(embeddings_list)
    print(f"Embeddings shape: {all_embeddings.shape}")

    print("Building FAISS index...")
    index = build_index(all_embeddings)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "faiss.index"
    faiss.write_index(index, str(index_path))

    if args.save_embeddings:
        np.save(output_dir / "embeddings.npy", all_embeddings)

    metadata_out = {
        "num_vectors": int(all_embeddings.shape[0]),
        "vector_dim": int(all_embeddings.shape[1]),
        "normalize": normalize,
        "model": args.model,
        "pretrained": args.pretrained,
        "checkpoint": str(checkpoint_path),
        "font_to_id": font_to_id,
        "samples": metadata_list,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, indent=2)

    print("\n✅ Phase 3 complete!")
    print(f"FAISS index saved to: {index_path}")
    print(f"Metadata saved to: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
