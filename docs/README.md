# Font Detection MVP

An AI-powered font detection application that identifies fonts from images using deep learning embeddings and vector similarity search.

## Project Structure

```
projects/
├── MVP_PLAN.md           # Detailed MVP plan and phases
├── requirements.txt       # Python dependencies
├── data_collection.py    # Phase 1: Font dataset collection
├── font.py               # (Placeholder - will be used later)
└── README.md             # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Check Available Fonts (Optional but Recommended)

Before running data collection, check which fonts are available on your system:

```bash
python check_fonts.py
```

This will show you:
- Which target fonts are available
- Which fonts need to be downloaded
- Recommendations based on availability

### 4. Run Phase 1: Data Collection

```bash
python data_collection.py
```

This will:
- Create a `font_dataset/` directory
- Generate font samples for 75 common fonts
- Save metadata in JSON format
- Create ~750 sample images (10 per font)

**Note**: If you don't have many fonts installed, the script will use default fonts as fallback. For best results, consider installing Google Fonts manually or we'll add API integration in a future update.

## Current Phase: Phase 1 - Data Collection

**Status**: In Progress

**What we're building**:
- Font dataset with 75 fonts
- 10 samples per font with varying text and sizes
- Organized directory structure for easy access

## Next Phases

See `MVP_PLAN.md` for the complete breakdown of all 7 phases.

## Notes

- The script uses system fonts for MVP (you may need to install Google Fonts manually)
- For production, integrate with Google Fonts API for automatic downloads
- Dataset will be used to fine-tune the embedding model in Phase 2

