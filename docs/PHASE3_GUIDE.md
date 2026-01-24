# Phase 3: Vector Database Setup Guide

## Overview

This phase generates embeddings for every font sample and builds a local FAISS index
for fast similarity search.

## Prerequisites

- ✅ Phase 1 complete (`font_dataset/` exists)
- ✅ Phase 2 complete (`models/best_model.pt` exists)
- ✅ Dependencies installed:
  ```powershell
  pip install -r requirements.txt
  ```

## Run Phase 3

```powershell
python phase3_vector_db.py
```

### Recommended (save embeddings for visualization)

```powershell
python phase3_vector_db.py --save_embeddings
```

### Optional flags

- Use a custom checkpoint:
  ```powershell
  python phase3_vector_db.py --checkpoint models/best_model.pt
  ```
- Save embeddings to a `.npy` file:
  ```powershell
  python phase3_vector_db.py --save_embeddings
  ```
- Use a different dataset folder:
  ```powershell
  python phase3_vector_db.py --dataset_dir font_dataset --metadata font_dataset/metadata.json
  ```

## Output Files

Generated in the `vector_db/` folder:

- `faiss.index` — FAISS vector index (cosine similarity via normalized vectors)
- `metadata.json` — sample metadata (font name, path, text, size)
- `embeddings.npy` — optional raw embeddings (only if `--save_embeddings` is used)

## Visualize Embeddings

```powershell
python visualize_embeddings.py --vector_db vector_db --method umap --output vector_db/embedding_plot.png
```

Options:
- `--method pca|tsne|umap`
- `--show` to open an interactive window
- `--max_points 2000` to subsample for large datasets

## Search a New Image

```powershell
python search_font.py --image path\\to\\query.png --vector_db vector_db --top_k 5
```

## Next Steps

- Phase 4: Build an image preprocessing pipeline for user uploads
- Add a search script that loads `faiss.index` and returns top-K font matches
