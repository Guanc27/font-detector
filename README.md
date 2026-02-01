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

An AI-powered font detection tool. Select text on any webpage using a Chrome extension, and the system identifies the font using deep learning embeddings and vector similarity search.

**Live API**: https://guanc27-check-fonts.hf.space

## How It Works

1. A Chrome extension lets you draw a rectangle around text on any webpage.
2. The selected region is screenshotted, cropped, and sent to a FastAPI backend.
3. The backend preprocesses the image (deskew, normalize, resize) and generates an embedding using a fine-tuned OpenCLIP ViT-B-32 model.
4. The embedding is compared against a FAISS index of 75 Google Fonts.
5. The top matches are returned with confidence scores and Google Fonts links.

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

## Setup

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

## Reproducing the Full Pipeline

The pipeline has six phases. If you just want to use the extension with the deployed API, skip to [Install the Chrome Extension](#install-the-chrome-extension).

### Phase 1: Data Collection

Download 75 Google Fonts and generate 20 training samples per font with varied text, sizes, and color combinations.

```bash
python download_google_fonts.py
python data_collection.py
```

This creates `font_dataset/` with 750+ labeled PNG images and a `metadata.json` mapping each image to its font.

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

No separate script to run. The `preprocessing.py` module is used at inference time to clean up real-world images before they reach the model:

- **Deskew**: Straighten rotated text using OpenCV contour analysis.
- **Normalize background**: Invert dark backgrounds, apply autocontrast.
- **Resize and pad**: Scale to 224x224 on a white canvas (avoids cropping wide text).
- **Text detection** (multi-region mode): EasyOCR locates text regions in full photos.

### Phase 5: API Server

The FastAPI server loads the model, FAISS index, and preprocessor at startup, then exposes two endpoints:

```bash
uvicorn api_server:app --reload --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server status, model info, index stats |
| `/detect` | POST | Upload an image, returns top-K font matches |

Test with curl:
```bash
curl -X POST http://localhost:8000/detect -F "file=@test.png"
```

Or open http://localhost:8000/docs for the interactive Swagger UI.

### Phase 6: Chrome Extension

The extension provides the browser UI for font detection.

#### Install the Chrome Extension

1. Open `chrome://extensions` in Chrome.
2. Enable **Developer mode** (toggle in top-right).
3. Click **Load unpacked** and select the `extension/` folder.
4. The Font Detector icon appears in your toolbar.

#### Usage

- **Right-click** anywhere on a webpage and select **Detect Font**.
- Draw a rectangle around the text you want to identify.
- Click the **Font Detector icon** in the toolbar to view results.

The extension defaults to the deployed API at `https://guanc27-check-fonts.hf.space`. To use a local server instead, click the extension icon, expand **Settings**, and change the API URL to `http://localhost:8000`.

## Deployment

The API is deployed as a Docker container on [Hugging Face Spaces](https://huggingface.co/spaces/Guanc27/check_fonts).

To deploy your own instance:

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

## API Reference

### POST /detect

**Request**:
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

## License

MIT License
