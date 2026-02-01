# Understanding: Dataset Generation vs Training

## The Key Point

**`train_embedding_model.py` does NOT use `data_collection.py` directly!**

It reads from `font_dataset/metadata.json` which was **already created** by `data_collection.py`.

---

## The Flow

```
Step 1: Download Fonts
get_all_google_fonts.py
    ↓
Downloads 1000 fonts to downloaded_fonts/
Saves list to downloaded_fonts_list.json

Step 2: Generate Dataset (THIS IS WHAT YOU'RE MISSING!)
data_collection.py
    ↓
Reads fonts from downloaded_fonts_list.json
Creates samples for each font
Saves to font_dataset/
Creates font_dataset/metadata.json (with 69 fonts if you used default)

Step 3: Train Model
train_embedding_model.py
    ↓
Reads font_dataset/metadata.json
Loads samples from font_dataset/samples/
Trains on whatever fonts are in metadata.json (69 fonts!)
```

---

## Why You're Seeing 69 Fonts

**The problem:**
- You downloaded 1000 fonts ✅
- But `font_dataset/metadata.json` was created with only 69 fonts ❌
- `train_embedding_model.py` reads from `metadata.json` → sees 69 fonts

**The solution:**
- You need to **regenerate** `font_dataset/` using `data_collection.py` with 1000 fonts
- This will create a NEW `metadata.json` with 1000 fonts
- Then `train_embedding_model.py` will see 1000 fonts

---

## What You Need to Do

### Step 1: Regenerate Dataset with 1000 Fonts

```python
# In Colab, after downloading fonts:
os.chdir('/content/drive/MyDrive/check_fonts')

# Regenerate dataset with all downloaded fonts
!python data_collection.py --num_fonts 1000 --num_samples 20
```

**This will:**
- Read `downloaded_fonts_list.json` (has 1000 fonts)
- Create samples for all 1000 fonts
- Save to `font_dataset/samples/`
- Create NEW `font_dataset/metadata.json` with 1000 fonts

### Step 2: Then Train

```python
# Now train_embedding_model.py will see 1000 fonts!
!python train_embedding_model.py \
  --epochs 20 \
  --batch_size 64 \
  --dataset_dir font_dataset \
  --save_dir models
```

---

## Complete Workflow

```python
# CELL 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# CELL 2: Navigate
import os
os.chdir('/content/drive/MyDrive/check_fonts')

# CELL 3: Download Fonts (if not done)
!pip install requests tqdm pillow
!python get_all_google_fonts.py --limit 1000

# CELL 4: Generate Dataset with 1000 Fonts ⭐ IMPORTANT!
!pip install pillow numpy tqdm
!python data_collection.py --num_fonts 1000 --num_samples 20
# This creates font_dataset/metadata.json with 1000 fonts

# CELL 5: Verify Dataset
import json
with open('font_dataset/metadata.json', 'r') as f:
    data = json.load(f)
    print(f"✅ Dataset has {data['num_fonts']} fonts")
    print(f"✅ Total samples: {sum(f['num_samples'] for f in data['fonts'])}")

# CELL 6: Train Model
!pip install torch torchvision open-clip-torch
!python train_embedding_model.py \
  --epochs 20 \
  --batch_size 64 \
  --dataset_dir font_dataset \
  --save_dir models
# Now it will train on 1000 fonts!
```

---

## Why This Happens

### `train_embedding_model.py` reads from `metadata.json`:

```python
# Line 108-111 in train_embedding_model.py
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

print(f"Metadata loaded: {len(metadata.get('fonts', []))} fonts")
```

**It doesn't care about:**
- How many fonts you downloaded
- `downloaded_fonts_list.json`
- `data_collection.py`

**It only cares about:**
- What's in `font_dataset/metadata.json`
- What samples exist in `font_dataset/samples/`

---

## The Missing Step

**You skipped regenerating the dataset!**

After downloading 1000 fonts, you need to:
1. ✅ Download fonts → `downloaded_fonts/` (done)
2. ❌ **Generate dataset** → `font_dataset/` (missing!)
3. ✅ Train model → reads from `font_dataset/` (but only sees old 69 fonts)

---

## Quick Fix

```python
# Regenerate dataset with 1000 fonts
!python data_collection.py --num_fonts 1000 --num_samples 20

# Verify it worked
import json
with open('font_dataset/metadata.json', 'r') as f:
    data = json.load(f)
    print(f"Fonts in dataset: {data['num_fonts']}")  # Should be 1000!

# Now train
!python train_embedding_model.py --epochs 20 --batch_size 64
```

---

## Summary

**The issue:**
- `train_embedding_model.py` reads from `font_dataset/metadata.json`
- Your `metadata.json` has 69 fonts (old dataset)
- You downloaded 1000 fonts but didn't regenerate the dataset

**The fix:**
- Run `data_collection.py --num_fonts 1000` to regenerate dataset
- This creates NEW `metadata.json` with 1000 fonts
- Then `train_embedding_model.py` will see 1000 fonts

**Remember:** Downloading fonts ≠ Generating dataset! You need both steps! 🎯

