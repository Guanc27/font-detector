# Fix: Where Files Should Be in Colab

## Your Situation

You ran:
```python
os.chdir('/content/drive/MyDrive')  # Changed to MyDrive
!python get_all_google_fonts.py     # Ran script
```

**Problem:**
- Script ran from `/content/drive/MyDrive/`
- Fonts downloaded to `/content/drive/MyDrive/downloaded_fonts/`
- But your project is in `/content/drive/MyDrive/check_fonts/`

---

## Where Files Should Be

### ✅ Correct Setup:

```
/content/drive/MyDrive/check_fonts/
├── get_all_google_fonts.py          ← Should be here
├── download_google_fonts.py         ← Should be here (needed for import)
├── train_embedding_model.py
├── data_collection.py
├── font_dataset/
└── downloaded_fonts/                ← Fonts should download here
```

### ❌ What Happened:

```
/content/drive/MyDrive/
├── get_all_google_fonts.py          ← You ran it from here
└── downloaded_fonts/                ← Fonts downloaded here (wrong location!)

/content/drive/MyDrive/check_fonts/
├── train_embedding_model.py
└── font_dataset/
```

---

## What to Do

### Option 1: Move Files to `check_fonts` Folder (Recommended)

**Step 1: Check where fonts downloaded**
```python
# In Colab, check:
!ls /content/drive/MyDrive/downloaded_fonts/ | head -5
```

**Step 2: Move fonts to correct location**
```python
# Move fonts to check_fonts folder
import shutil
shutil.move('/content/drive/MyDrive/downloaded_fonts', 
            '/content/drive/MyDrive/check_fonts/downloaded_fonts')
```

**Step 3: Move scripts to check_fonts**
```python
# Move scripts to check_fonts folder
import shutil
shutil.move('/content/drive/MyDrive/get_all_google_fonts.py', 
            '/content/drive/MyDrive/check_fonts/get_all_google_fonts.py')
shutil.move('/content/drive/MyDrive/download_google_fonts.py', 
            '/content/drive/MyDrive/check_fonts/download_google_fonts.py')
```

**Step 4: Navigate and verify**
```python
os.chdir('/content/drive/MyDrive/check_fonts')
!ls -la  # Should see both scripts and downloaded_fonts/
```

---

### Option 2: Re-run from Correct Location

**Step 1: Make sure both scripts are in `check_fonts`**
```python
# Upload or move both files to check_fonts folder
os.chdir('/content/drive/MyDrive/check_fonts')
!ls get_all_google_fonts.py download_google_fonts.py  # Should exist
```

**Step 2: Re-run download**
```python
os.chdir('/content/drive/MyDrive/check_fonts')
!python get_all_google_fonts.py --limit 1000
```

**Note:** Script will skip fonts that already exist, so it's safe to re-run!

---

## Why It Matters

### Font Download Location

The script downloads fonts to `downloaded_fonts/` **relative to current directory**:

```python
# In download_google_fonts.py line 24:
self.output_dir = Path(output_dir)  # Default: "downloaded_fonts"
```

So:
- Run from `/content/drive/MyDrive/` → Downloads to `/content/drive/MyDrive/downloaded_fonts/`
- Run from `/content/drive/MyDrive/check_fonts/` → Downloads to `/content/drive/MyDrive/check_fonts/downloaded_fonts/`

### Import Dependency

`get_all_google_fonts.py` imports from `download_google_fonts.py`:
```python
from download_google_fonts import GoogleFontsDownloader
```

**Both files must be in the same directory** for the import to work!

---

## Recommended Solution

### Complete Setup in Colab:

```python
# CELL 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# CELL 2: Navigate to check_fonts
import os
os.chdir('/content/drive/MyDrive/check_fonts')
print(f"Current directory: {os.getcwd()}")

# CELL 3: Verify files are there
!ls -la
# Should see:
# - get_all_google_fonts.py
# - download_google_fonts.py
# - train_embedding_model.py
# - font_dataset/

# CELL 4: Download fonts (if not already done)
!pip install requests tqdm pillow
!python get_all_google_fonts.py --limit 1000
# Fonts will download to: /content/drive/MyDrive/check_fonts/downloaded_fonts/

# CELL 5: Verify fonts downloaded
!ls downloaded_fonts/ | head -10

# CELL 6: Run training
!python train_embedding_model.py \
  --epochs 20 \
  --batch_size 64 \
  --dataset_dir font_dataset \
  --save_dir models
```

---

## Quick Fix Commands

If fonts downloaded to wrong location, move them:

```python
import shutil
import os

# Navigate to check_fonts
os.chdir('/content/drive/MyDrive/check_fonts')

# Check if fonts exist in MyDrive
if os.path.exists('/content/drive/MyDrive/downloaded_fonts'):
    # Move fonts to check_fonts
    if os.path.exists('downloaded_fonts'):
        # Merge folders
        import shutil
        for font_dir in os.listdir('/content/drive/MyDrive/downloaded_fonts'):
            src = f'/content/drive/MyDrive/downloaded_fonts/{font_dir}'
            dst = f'downloaded_fonts/{font_dir}'
            if not os.path.exists(dst):
                shutil.move(src, dst)
    else:
        # Move entire folder
        shutil.move('/content/drive/MyDrive/downloaded_fonts', 'downloaded_fonts')
    
    print("✅ Fonts moved to check_fonts/downloaded_fonts/")
else:
    print("Fonts not found in MyDrive - they may already be in check_fonts/")
```

---

## Summary

**What happened:**
- You ran script from `/content/drive/MyDrive/`
- Fonts downloaded to `/content/drive/MyDrive/downloaded_fonts/`
- But your project expects them in `/content/drive/MyDrive/check_fonts/downloaded_fonts/`

**What to do:**
1. ✅ Put both scripts (`get_all_google_fonts.py` and `download_google_fonts.py`) in `check_fonts` folder
2. ✅ Move fonts from `MyDrive/downloaded_fonts/` to `check_fonts/downloaded_fonts/`
3. ✅ Or re-run script from `check_fonts` folder (will skip existing fonts)

**Best practice:** Always run scripts from the `check_fonts` folder! 🎯

