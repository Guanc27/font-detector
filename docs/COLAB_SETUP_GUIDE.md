# Google Colab Setup Guide - Step by Step

## Why Google Drive?

### The Key Issue: Colab Runs in the Cloud

**Important**: Colab runs on Google's servers, **not on your computer**.

This means:
- ❌ Your local files (`C:\Users\user\...`) are **NOT accessible** to Colab
- ❌ Colab can't see files on your laptop
- ✅ You need to **upload** your files to Colab's cloud environment

### Two Ways to Get Files into Colab

**Option 1: Google Drive** (Recommended)
- Upload files to Google Drive
- Mount Drive in Colab
- Access files from Colab

**Option 2: Direct Upload**
- Upload files directly to Colab session
- Files deleted when session ends
- Good for quick tests

---

## What Files Do You Need?

### Required Files

You need **more than just** `train_embedding_model.py`:

1. **Code files**:
   - `train_embedding_model.py` ✅
   - `data_collection.py` (if you need to regenerate data)
   - Any other Python scripts you use

2. **Dataset**:
   - `font_dataset/` folder (with all samples)
   - `font_dataset/metadata.json`

3. **Dependencies**:
   - `requirements.txt` (for installing packages)

### File Structure in Colab

```
/content/drive/MyDrive/check_fonts/  (or /content/check_fonts/)
├── train_embedding_model.py
├── data_collection.py
├── requirements.txt
├── font_dataset/
│   ├── metadata.json
│   └── samples/
│       ├── Roboto/
│       │   ├── sample_000.png
│       │   └── ...
│       └── ...
└── models/  (created during training)
```

---

## Method 1: Using Google Drive (Recommended)

### Step 1: Prepare Your Files Locally

**Option A: Upload Entire Folder**
1. Zip your `check_fonts` folder:
   ```powershell
   # In PowerShell, navigate to projects folder
   Compress-Archive -Path check_fonts -DestinationPath check_fonts.zip
   ```

2. Upload `check_fonts.zip` to Google Drive

3. In Colab, extract it:
   ```python
   !unzip /content/drive/MyDrive/check_fonts.zip -d /content/
   ```

**Option B: Upload Individual Files**
1. Upload these to Google Drive:
   - `train_embedding_model.py`
   - `font_dataset/` folder (entire folder)
   - `requirements.txt` (optional)

### Step 2: Colab Setup Code

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Navigate to your project
import os
os.chdir('/content/drive/MyDrive/check_fonts')  # Adjust path to where you uploaded

# Cell 3: Verify files are there
!ls -la
!ls font_dataset/samples/ | head -5  # Check dataset exists

# Cell 4: Install dependencies
!pip install torch torchvision open-clip-torch tqdm pillow numpy

# Cell 5: Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 6: Run training
!python train_embedding_model.py \
  --epochs 10 \
  --batch_size 32 \
  --dataset_dir font_dataset \
  --save_dir models
```

---

## Method 2: Direct Upload (Quick Test)

### Step 1: Upload Files Directly to Colab

```python
# Cell 1: Install dependencies
!pip install torch torchvision open-clip-torch tqdm pillow numpy

# Cell 2: Upload files using Colab's file uploader
from google.colab import files
uploaded = files.upload()  # Select train_embedding_model.py

# Cell 3: Upload dataset folder (you'll need to zip it first)
# Upload font_dataset.zip, then:
!unzip font_dataset.zip

# Cell 4: Run training
!python train_embedding_model.py --epochs 10 --batch_size 32
```

**Note**: Files uploaded this way are **deleted when session ends**!

---

## Method 3: GitHub (Best for Code)

### Step 1: Push Code to GitHub

```powershell
# In your local check_fonts folder
git init
git add train_embedding_model.py data_collection.py requirements.txt
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/check_fonts.git
git push -u origin main
```

### Step 2: Clone in Colab

```python
# Cell 1: Clone repository
!git clone https://github.com/yourusername/check_fonts.git
%cd check_fonts

# Cell 2: Upload dataset (still need to upload dataset separately)
from google.colab import files
# Upload font_dataset.zip, then:
!unzip font_dataset.zip

# Cell 3: Install dependencies
!pip install torch torchvision open-clip-torch tqdm pillow numpy

# Cell 4: Run training
!python train_embedding_model.py --epochs 10 --batch_size 32
```

---

## Complete Colab Notebook Example

Here's a complete notebook you can copy-paste:

```python
# ============================================
# CELL 1: Setup and Mount Drive
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================
# CELL 2: Navigate to Project
# ============================================
import os
# Change this to where you uploaded your files
project_path = '/content/drive/MyDrive/check_fonts'
os.chdir(project_path)
print(f"Current directory: {os.getcwd()}")

# ============================================
# CELL 3: Verify Files Exist
# ============================================
!ls -la
print("\nChecking dataset...")
!ls font_dataset/samples/ | head -5

# ============================================
# CELL 4: Install Dependencies
# ============================================
!pip install torch torchvision open-clip-torch tqdm pillow numpy

# ============================================
# CELL 5: Check GPU
# ============================================
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================
# CELL 6: Download Fonts (Optional - if needed)
# ============================================
# Only run this if you need to download more fonts
# Fonts will be saved to downloaded_fonts/ in Drive
!pip install requests tqdm pillow
!python get_all_google_fonts.py --limit 1000

# ============================================
# CELL 7: Generate Dataset (Optional - if needed)
# ============================================
# Only run this if you need to regenerate samples from downloaded fonts
!python data_collection.py --num_fonts 1000 --num_samples 20

# ============================================
# CELL 8: Run Training
# ============================================
!python train_embedding_model.py \
  --epochs 10 \
  --batch_size 32 \
  --dataset_dir font_dataset \
  --metadata font_dataset/metadata.json \
  --save_dir models

# ============================================
# CELL 9: Download Model (Optional)
# ============================================
from google.colab import files
files.download('models/best_model.pt')
```

---

## Step-by-Step: First Time Setup

### Step 1: Prepare Files Locally

1. **Make sure your dataset is ready**:
   ```powershell
   # Check dataset exists
   dir font_dataset\samples
   ```

2. **Zip your project** (optional but easier):
   ```powershell
   Compress-Archive -Path check_fonts -DestinationPath check_fonts.zip
   ```

### Step 2: Upload to Google Drive

1. Go to https://drive.google.com/
2. Click "New" → "File upload"
3. Upload:
   - `check_fonts.zip` (if you zipped it), OR
   - Individual files: `train_embedding_model.py`, `font_dataset/` folder
4. Note the path (usually `/content/drive/MyDrive/check_fonts`)

### Step 3: Create Colab Notebook

1. Go to https://colab.research.google.com/
2. Click "New notebook"
3. Copy-paste the notebook code above
4. **Change the path** in Cell 2 to match where you uploaded files

### Step 4: Enable GPU

1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Save

### Step 5: Run Cells

1. Run cells one by one (Shift+Enter)
2. For Cell 1 (Drive mount): Click the link, authorize, copy code
3. Watch training progress!

---

## What Gets Uploaded?

### Minimum Required:

```
check_fonts/
├── train_embedding_model.py  ✅ Required
├── font_dataset/              ✅ Required
│   ├── metadata.json
│   └── samples/
│       └── [all font folders]
└── requirements.txt           ⚠️ Optional (can install manually)
```

### Nice to Have:

```
check_fonts/
├── data_collection.py         (if you need to regenerate)
├── download_google_fonts.py   (if you need to download fonts)
└── docs/                      (documentation, not needed for training)
```

---

## Common Issues and Solutions

### Issue: "File not found"

**Problem**: Colab can't find your files

**Solution**:
```python
# Check where files actually are
!ls /content/drive/MyDrive/
# Find your folder, then update path
os.chdir('/content/drive/MyDrive/YOUR_FOLDER_NAME')
```

### Issue: "Dataset not found"

**Problem**: `font_dataset` folder not uploaded correctly

**Solution**:
```python
# Check if dataset exists
!ls -la font_dataset/
!ls font_dataset/samples/ | head -5

# If not found, upload it:
from google.colab import files
# Upload font_dataset.zip, then:
!unzip font_dataset.zip
```

### Issue: "Module not found"

**Problem**: Missing dependencies

**Solution**:
```python
!pip install torch torchvision open-clip-torch tqdm pillow numpy
```

### Issue: "Out of memory"

**Problem**: Batch size too large

**Solution**:
```python
# Reduce batch size
!python train_embedding_model.py --batch_size 16
```

---

## File Size Considerations

### Dataset Size

Your `font_dataset` folder might be large:
- 75 fonts × 20 samples = ~1,500 images
- Each image ~50-100 KB
- Total: ~75-150 MB

**Upload options**:
1. **Google Drive**: Handles large files well
2. **Direct upload**: May timeout for large files
3. **GitHub**: Not good for large datasets (use Git LFS or separate upload)

### Recommended Approach

1. **Code**: Upload to GitHub (easy to update)
2. **Dataset**: Upload to Google Drive (large files)
3. **In Colab**: Clone code, mount Drive for dataset

---

## Quick Reference

### Files Needed:
- ✅ `train_embedding_model.py` (code)
- ✅ `font_dataset/` (data)
- ⚠️ `requirements.txt` (optional)

### Why Google Drive?
- Colab runs in cloud → can't access local files
- Drive = persistent storage in cloud
- Mount Drive = access files from Colab

### Upload Methods:
1. **Google Drive** (recommended) - persistent
2. **Direct upload** - temporary, deleted when session ends
3. **GitHub** - good for code, not for large datasets

---

## Summary

**Q: Why Google Drive if fonts are local?**  
A: Colab runs in the cloud and can't access your local files. Drive stores files in the cloud so Colab can access them.

**Q: How do I upload code?**  
A: Upload `train_embedding_model.py` AND `font_dataset/` folder to Google Drive, then mount Drive in Colab.

**Q: Is it just train_embedding_model.py?**  
A: No! You need:
- ✅ `train_embedding_model.py` (code)
- ✅ `font_dataset/` folder (your dataset)
- ⚠️ Dependencies installed via pip

The easiest way: Upload your entire `check_fonts` folder to Google Drive, then mount and run!




