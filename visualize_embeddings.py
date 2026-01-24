"""
Visualize font embeddings from Phase 3 using PCA/TSNE/UMAP.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required. Install it with: pip install matplotlib"
    ) from exc

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required. Install it with: pip install scikit-learn"
    ) from exc

try:
    import umap  # type: ignore
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(vector_db_dir):
    embeddings_path = Path(vector_db_dir) / "embeddings.npy"
    if embeddings_path.exists():
        return np.load(embeddings_path)

    index_path = Path(vector_db_dir) / "faiss.index"
    if not index_path.exists():
        raise FileNotFoundError(
            "No embeddings.npy or faiss.index found. "
            "Run Phase 3 with --save_embeddings or ensure faiss.index exists."
        )

    if not HAS_FAISS:
        raise ImportError(
            "faiss-cpu is required to reconstruct embeddings from faiss.index. "
            "Install it with: pip install faiss-cpu"
        )

    index = faiss.read_index(str(index_path))
    if not hasattr(index, "reconstruct"):
        raise RuntimeError("FAISS index does not support vector reconstruction.")

    vectors = np.zeros((index.ntotal, index.d), dtype="float32")
    for i in range(index.ntotal):
        vectors[i] = index.reconstruct(i)
    return vectors


def choose_method(method):
    if method == "umap":
        if not HAS_UMAP:
            raise ImportError(
                "umap-learn is required for --method umap. "
                "Install it with: pip install umap-learn"
            )
        return "umap"
    if method in {"pca", "tsne"}:
        return method
    raise ValueError("method must be one of: pca, tsne, umap")


def reduce_embeddings(embeddings, method, seed):
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        return reducer.fit_transform(embeddings)
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=seed, init="pca")
        return reducer.fit_transform(embeddings)
    reducer = umap.UMAP(n_components=2, random_state=seed)
    return reducer.fit_transform(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Visualize font embeddings")
    parser.add_argument("--vector_db", type=str, default="vector_db",
                        help="Path to vector_db directory")
    parser.add_argument("--method", type=str, default="umap",
                        help="Dimensionality reduction: pca, tsne, umap")
    parser.add_argument("--output", type=str, default="embedding_plot.png",
                        help="Output image path")
    parser.add_argument("--show", action="store_true",
                        help="Display plot interactively")
    parser.add_argument("--max_points", type=int, default=2000,
                        help="Max points to plot (randomly sampled)")
    parser.add_argument("--legend_max", type=int, default=12,
                        help="Show legend only if font count <= legend_max")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling and reducers")
    args = parser.parse_args()

    vector_db_dir = Path(args.vector_db)
    metadata_path = vector_db_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    metadata = load_metadata(metadata_path)
    samples = metadata.get("samples", [])

    embeddings = load_embeddings(vector_db_dir)
    if len(samples) != embeddings.shape[0]:
        raise RuntimeError(
            f"Metadata samples ({len(samples)}) do not match embeddings "
            f"({embeddings.shape[0]}). Re-run Phase 3 with --save_embeddings."
        )

    rng = np.random.default_rng(args.seed)
    total = embeddings.shape[0]
    if total > args.max_points:
        idx = rng.choice(total, size=args.max_points, replace=False)
        embeddings = embeddings[idx]
        samples = [samples[i] for i in idx]

    method = choose_method(args.method)
    reduced = reduce_embeddings(embeddings, method, args.seed)

    font_names = [s["font_name"] for s in samples]
    unique_fonts = sorted(set(font_names))
    font_to_color = {name: i for i, name in enumerate(unique_fonts)}
    colors = [font_to_color[name] for name in font_names]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, cmap="tab20", s=12, alpha=0.8)
    plt.title(f"Font Embeddings ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    if len(unique_fonts) <= args.legend_max:
        handles, _ = scatter.legend_elements(num=len(unique_fonts))
        plt.legend(handles, unique_fonts, title="Fonts", loc="best", fontsize=8)

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
