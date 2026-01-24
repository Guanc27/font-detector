# Colab Notebook: Where to Run Font Download

## Complete Colab Notebook Structure

Here's the **exact order** of cells you should use:

```python
# ============================================
# CELL 1: Mount Google Drive
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================
# CELL 2: Navigate to Project
# ============================================
import os
project_path = '/content/drive/MyDrive/check_fonts'  # Adjust to your path
os.chdir(project_path)
print(f"Current directory: {os.getcwd()}")

# ============================================
# CELL 3: Upload Scripts (if not already in Drive)
# ============================================
# ⚠️ IMPORTANT: Colab runs in the cloud and can't see your local files!
# You need to get the files INTO Colab first before running them.

# Option A: Upload via Colab file uploader (if files NOT in Drive)
from google.colab import files
uploaded = files.upload()  # Select get_all_google_fonts.py and download_google_fonts.py
# This uploads files from YOUR COMPUTER to Colab's cloud environment

# Option B: If files are already in Drive, skip this cell
# (Files are already accessible after mounting Drive in Cell 1)

# ============================================
# CELL 4: Download Fonts ⭐ THIS IS WHERE YOU RUN IT
# ============================================
# Install dependencies and download fonts in one cell
!pip install requests tqdm pillow
!python get_all_google_fonts.py --limit 1000

# This will:
# - Download fonts to downloaded_fonts/
# - Save list to downloaded_fonts_list.json
# - Skip fonts that already exist (won't re-download)

# ============================================
# CELL 5: Verify Fonts Downloaded
# ============================================
!ls downloaded_fonts/ | head -10
print(f"\nTotal font folders: {len([d for d in os.listdir('downloaded_fonts') if os.path.isdir(f'downloaded_fonts/{d}')])}")

# ============================================
# CELL 6: Generate Dataset (if needed)
# ============================================
# Only run this if you need to regenerate samples
!pip install torch torchvision open-clip-torch tqdm pillow numpy
!python data_collection.py --num_fonts 1000 --num_samples 20

# ============================================
# CELL 7: Check GPU
# ============================================
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================
# CELL 8: Run Training
# ============================================
!python train_embedding_model.py \
  --epochs 20 \
  --batch_size 64 \
  --dataset_dir font_dataset \
  --save_dir models
```

---

## Quick Answer: Which Cell?

**Run `get_all_google_fonts.py` in Cell 4** (after mounting Drive, install dependencies in the same cell)

---

## Step-by-Step Instructions

### Step 1: Setup (Cells 1-3)
1. **Cell 1**: Mount Google Drive
2. **Cell 2**: Navigate to your project folder
3. **Cell 3**: Upload scripts (if needed)

### Step 2: Download Fonts (Cell 4) ⭐
```python
# Install dependencies and download fonts
!pip install requests tqdm pillow
!python get_all_google_fonts.py --limit 1000
```

**What this does:**
- Downloads 1000 fonts to `downloaded_fonts/`
- Takes ~30-60 minutes
- Skips fonts that already exist
- Saves list to `downloaded_fonts_list.json`

**Options:**
```python
# Download all fonts (default: 1000)
!python get_all_google_fonts.py

# Download specific number
!python get_all_google_fonts.py --limit 500

# Resume from font 500
!python get_all_google_fonts.py --limit 500 --start_from 500
```

### Step 3: Verify (Cell 5)
Check that fonts downloaded successfully

### Step 4: Generate Dataset (Cell 6)
Only if you need to create samples from downloaded fonts

### Step 5: Train (Cells 7-8)
Check GPU and run training

---

## Important Notes

### ⚠️ Font Download Takes Time
- **1000 fonts**: ~30-60 minutes
- **500 fonts**: ~15-30 minutes
- Run this **once**, then fonts stay in Drive

### ✅ Fonts Persist in Drive
- Fonts are saved to `downloaded_fonts/` in Drive
- They persist between Colab sessions
- No need to re-download if already there

### 🔄 Skip Already Downloaded Fonts
- Script automatically skips existing fonts
- Safe to re-run if interrupted
- Won't waste time re-downloading

---

## Troubleshooting

### "File not found: get_all_google_fonts.py"
**Solution**: Make sure you uploaded the script to Drive, or upload it in Cell 3

### "Module not found: requests"
**Solution**: Run Cell 4 to install dependencies first

### "Download interrupted"
**Solution**: Just re-run Cell 5 - it will skip already downloaded fonts and continue

### "Out of disk space"
**Solution**: 
- Download fewer fonts: `--limit 500`
- Or download in batches: `--limit 500 --start_from 0`, then `--limit 500 --start_from 500`

---

## Recommended Workflow

### First Time Setup:
1. Upload `get_all_google_fonts.py` and `download_google_fonts.py` to Drive
2. Run Cells 1-5 (download fonts)
3. Wait for download to complete (~30-60 min)
4. Run Cell 7 (generate dataset)
5. Run Cells 8-9 (train model)

### Subsequent Sessions:
1. Run Cells 1-2 (mount Drive, navigate)
2. Skip Cell 5 (fonts already downloaded)
3. Run Cells 8-9 (train model)

---

## Example: Minimal Notebook

If you just want to download fonts quickly:

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Navigate
import os
os.chdir('/content/drive/MyDrive/check_fonts')

# Cell 3: Download Fonts (installs dependencies and downloads)
!pip install requests tqdm pillow
!python get_all_google_fonts.py --limit 1000
```

That's it! Fonts will be saved to Drive and ready for next time.

