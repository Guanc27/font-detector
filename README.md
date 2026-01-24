# Font Detection MVP

An AI-powered font detection application that identifies fonts from images using deep learning embeddings and vector similarity search.

## Quick Start

1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Download fonts:**
   ```powershell
   python download_google_fonts.py
   ```

3. **Generate dataset:**
   ```powershell
   python data_collection.py
   ```

4. **Train model:**
   ```powershell
   python train_embedding_model.py
   ```

5. **Build vector database (Phase 3):**
   ```powershell
   python phase3_vector_db.py
   ```

6. **Search a new image:**
   ```powershell
   python search_font.py --image path\\to\\query.png --top_k 5
   ```

## Documentation

All documentation is available in the [`docs/`](docs/) folder:

- **[MVP_PLAN.md](docs/MVP_PLAN.md)** - Complete project plan and phases
- **[SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md)** - Installation guide
- **[QUICK_START.md](docs/QUICK_START.md)** - Quick start guide for downloading fonts
- **[PHASE2_GUIDE.md](docs/PHASE2_GUIDE.md)** - Model training guide
- **[PHASE3_GUIDE.md](docs/PHASE3_GUIDE.md)** - Vector database setup guide
- **[MODEL_ACCURACY_GUIDE.md](docs/MODEL_ACCURACY_GUIDE.md)** - Accuracy expectations and improvements
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[FONT_EXPLANATION.md](docs/FONT_EXPLANATION.md)** - Understanding system fonts
- **[API_EXPLANATION.md](docs/API_EXPLANATION.md)** - Google Fonts API explanation
- **[DOWNLOAD_METHODS_EXPLANATION.md](docs/DOWNLOAD_METHODS_EXPLANATION.md)** - Font download methods

## Project Structure

```
check_fonts/
├── docs/                    # All documentation
├── downloaded_fonts/        # Downloaded Google Fonts
├── font_dataset/            # Generated font samples
├── models/                  # Trained models (after Phase 2)
├── data_collection.py       # Phase 1: Dataset generation
├── download_google_fonts.py # Font downloader
├── train_embedding_model.py # Phase 2: Model training
├── phase3_vector_db.py      # Phase 3: Vector database setup
├── search_font.py           # Query the vector database
└── requirements.txt         # Python dependencies
```

## Current Status

- ✅ Phase 1: Data Collection - Complete
- ✅ Phase 2: Model Training - Ready
- ✅ Phase 3: Vector Database - Ready
- ⏳ Phase 4: Image Processing - Pending
- ⏳ Phase 5: LLM Integration - Pending
- ⏳ Phase 6: UI Development - Pending

## License

MIT License

