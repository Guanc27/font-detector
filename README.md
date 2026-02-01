---
title: Check Fonts
emoji: 🔤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: 'Detects fonts from an image. '
---

# Font Detector

An AI-powered font detection tool. Select text on any webpage using a Chrome extension, and the system identifies the font using deep learning embeddings and vector similarity search. The backend is already deployed -- just install the extension and start using it.

**Live API**: https://guanc27-check-fonts.hf.space

## Install the Chrome Extension

1. Clone or download this repository.
2. Open `chrome://extensions` in Chrome.
3. Enable **Developer mode** (toggle in top-right).
4. Click **Load unpacked** and select the `extension/` folder.
5. The Font Detector icon appears in your toolbar.

## Usage

- **Right-click** anywhere on a webpage and select **Detect Font**.
- Draw a rectangle around the text you want to identify.
- Click the **Font Detector icon** in the toolbar to view results.

The extension connects to a hosted API by default. No server setup required.

## How It Works

1. The Chrome extension screenshots the selected region and sends it to a FastAPI backend hosted on [Hugging Face Spaces](https://huggingface.co/spaces/Guanc27/check_fonts).
2. The backend preprocesses the image (deskew, normalize, resize) and generates an embedding using a fine-tuned OpenCLIP ViT-B-32 model.
3. The embedding is compared against a FAISS index of 75 Google Fonts using cosine similarity.
4. The top matches are returned with confidence scores and Google Fonts links.

## API Reference

The API is public. You can call it directly without the extension.

### POST /detect

```bash
curl -X POST https://guanc27-check-fonts.hf.space/detect -F "file=@image.png"
```

**Parameters**:
- `file` (form-data): Image file (PNG, JPEG, WebP)
- `top_k` (query, optional): Number of results, 1-50 (default: 5)
- `mode` (query, optional): `"single"` (pre-cropped region) or `"multi"` (full photo with text detection)

**Response**:
```json
{
  "matches": [
    {
      "rank": 1,
      "font_name": "Roboto",
      "score": 0.8234,
      "google_fonts_url": "https://fonts.google.com/specimen/Roboto"
    }
  ],
  "processing_time_ms": 145.3,
  "regions_detected": 1
}
```

Interactive docs are available at https://guanc27-check-fonts.hf.space/docs.

## Project Structure

```
check_fonts/
|-- api_server.py              # FastAPI backend (deployed on HF Spaces)
|-- preprocessing.py           # Image cleanup pipeline
|-- data_collection.py         # Phase 1: generate training images
|-- download_google_fonts.py   # Download font files from Google Fonts
|-- train_embedding_model.py   # Phase 2: fine-tune OpenCLIP
|-- phase3_vector_db.py        # Phase 3: build FAISS vector index
|-- search_font.py             # CLI tool to query fonts from an image
|-- Dockerfile                 # Container for deployment
|-- requirements.txt           # Python dependencies
|-- models/
|   |-- best_model2.pt         # Trained model checkpoint
|-- vector_db/
|   |-- faiss.index            # FAISS similarity search index
|   |-- metadata.json          # Font name mappings
|-- extension/
|   |-- manifest.json          # Chrome extension config
|   |-- background.js          # Screenshot, crop, API call logic
|   |-- content.js             # Selection rectangle overlay
|   |-- popup.html/css/js      # Extension popup UI
```

---

## Development

Everything below is for developers who want to rebuild the model, modify the pipeline, or deploy their own instance.

### Prerequisites

- Python 3.11+
- Git with [Git LFS](https://git-lfs.com/) installed

### Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Phase 1: Data Collection

Download 75 Google Fonts and generate 20 training samples per font with varied text, sizes, and color combinations.

```bash
python download_google_fonts.py
python data_collection.py
```

Creates `font_dataset/` with 750+ labeled PNG images and a `metadata.json` mapping each image to its font.

### Phase 2: Model Training

Fine-tune an OpenCLIP ViT-B-32 model on the font dataset. The pretrained model (trained on 400M image-text pairs) is adapted by unfreezing the last few transformer layers and adding a classification head for 75 font classes.

Training applies data augmentation (rotation, blur, noise, perspective transforms) to improve robustness to real-world screenshots.

```bash
python train_embedding_model.py
```

Saves `models/best_model2.pt` (~687 MB). Training parameters: 10 epochs, batch size 32, Adam optimizer with lr=1e-4, cross-entropy loss, 70/15/15 train/val/test split.

### Phase 3: Vector Database

Generate embeddings for every training sample and build a FAISS index for fast nearest-neighbor search.

```bash
python phase3_vector_db.py
```

Saves `vector_db/faiss.index` and `vector_db/metadata.json`. Uses `IndexFlatIP` with L2-normalized vectors (equivalent to cosine similarity search).

### Phase 4: Image Preprocessing

No separate script to run. The `preprocessing.py` module is used at inference time to clean up real-world images:

- **Deskew**: Straighten rotated text using OpenCV contour analysis.
- **Normalize background**: Invert dark backgrounds, apply autocontrast.
- **Resize and pad**: Scale to 224x224 on a white canvas (avoids cropping wide text).
- **Text detection** (multi-region mode): EasyOCR locates text regions in full photos.

### Phase 5: Run the API Server Locally

```bash
uvicorn api_server:app --reload --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server status, model info, index stats |
| `/detect` | POST | Upload an image, returns top-K font matches |

Open http://localhost:8000/docs for the interactive Swagger UI. To point the extension at your local server, click the extension icon, expand **Settings**, and change the API URL to `http://localhost:8000`.

### Deploy Your Own Instance

The API is deployed as a Docker container on [Hugging Face Spaces](https://huggingface.co/spaces/Guanc27/check_fonts).

1. Create a new Space on [huggingface.co](https://huggingface.co/new-space) with the Docker SDK.
2. Add the Space as a git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<username>/<space-name>
   ```
3. Push:
   ```bash
   git push hf main
   ```

The `Dockerfile` handles everything: installs dependencies, pre-downloads EasyOCR models, copies the trained model and vector database, and starts Uvicorn on port 7860.

## License

MIT License
