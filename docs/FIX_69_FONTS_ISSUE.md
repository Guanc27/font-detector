# Fix: Still Seeing 69 Fonts Instead of 1000

## The Problem

After downloading 1000 fonts, `data_collection.py` still only uses 69 fonts.

## Root Causes

### Issue 1: Default `num_fonts=75`

Even if `downloaded_fonts_list.json` has 1000 fonts, `data_collection.py` defaults to 75:

```python
# Line 368 in data_collection.py
num_fonts=args.num_fonts if args.num_fonts else 75  # ❌ Always defaults to 75!
```

**Solution:** Pass `--num_fonts` argument or let it auto-detect.

---

### Issue 2: `downloaded_fonts_list.json` Not Found

The script looks for `downloaded_fonts_list.json` in the **current directory**:

```python
# Line 66 in data_collection.py
downloaded_fonts_file = Path("downloaded_fonts_list.json")  # Relative path!
```

**If you ran `get_all_google_fonts.py` from wrong directory:**
- File might be in `/content/drive/MyDrive/downloaded_fonts_list.json`
- But script looks in `/content/drive/MyDrive/check_fonts/downloaded_fonts_list.json`

---

## Solutions

### Solution 1: Pass `--num_fonts` Argument (Quick Fix)

```python
# In Colab, when running data_collection.py:
!python data_collection.py --num_fonts 1000 --num_samples 20
```

This explicitly tells it to use 1000 fonts.

---

### Solution 2: Check File Location

```python
# Check if downloaded_fonts_list.json exists
import os
os.chdir('/content/drive/MyDrive/check_fonts')

# Check if file exists
!ls downloaded_fonts_list.json

# If not found, check parent directory
!ls /content/drive/MyDrive/downloaded_fonts_list.json

# If found in wrong location, move it
import shutil
if os.path.exists('/content/drive/MyDrive/downloaded_fonts_list.json'):
    shutil.move('/content/drive/MyDrive/downloaded_fonts_list.json', 
                'downloaded_fonts_list.json')
    print("✅ Moved downloaded_fonts_list.json to check_fonts/")
```

---

### Solution 3: Verify File Contents

```python
# Check what's in the JSON file
import json
with open('downloaded_fonts_list.json', 'r') as f:
    data = json.load(f)
    print(f"Total fonts in file: {data.get('total', 0)}")
    print(f"Available fonts: {len(data.get('available', []))}")
    print(f"First 10 fonts: {data.get('available', [])[:10]}")
```

---

### Solution 4: Use Updated Code (Auto-Detection)

I've updated `data_collection.py` to automatically detect downloaded fonts. After updating:

```python
# It will automatically use all downloaded fonts if file exists
!python data_collection.py --num_samples 20
# No need to specify --num_fonts, it will auto-detect!
```

---

## Step-by-Step Fix in Colab

### Step 1: Check Current Situation

```python
import os
os.chdir('/content/drive/MyDrive/check_fonts')

# Check if JSON file exists
!ls downloaded_fonts_list.json

# Check contents
import json
if os.path.exists('downloaded_fonts_list.json'):
    with open('downloaded_fonts_list.json', 'r') as f:
        data = json.load(f)
        print(f"✅ Found {len(data.get('available', []))} fonts in JSON file")
else:
    print("❌ downloaded_fonts_list.json not found!")
```

---

### Step 2: Fix File Location (if needed)

```python
# If file is in wrong location, move it
import shutil
if os.path.exists('/content/drive/MyDrive/downloaded_fonts_list.json'):
    shutil.move('/content/drive/MyDrive/downloaded_fonts_list.json', 
                'downloaded_fonts_list.json')
    print("✅ Moved file to check_fonts/")
```

---

### Step 3: Run with Explicit Font Count

```python
# Option A: Use all fonts from JSON (if file exists)
!python data_collection.py --num_fonts 1000 --num_samples 20

# Option B: Let it auto-detect (after code update)
!python data_collection.py --num_samples 20
```

---

## Why This Happens

### The Flow:

1. **`get_all_google_fonts.py`** downloads fonts
   - Saves to `downloaded_fonts/`
   - Saves list to `downloaded_fonts_list.json`
   - **Location depends on where you ran it!**

2. **`data_collection.py`** loads fonts
   - Looks for `downloaded_fonts_list.json` in **current directory**
   - If found, loads font list
   - **BUT** defaults to `num_fonts=75` if not specified

3. **Result:**
   - Even if 1000 fonts loaded, `num_fonts=75` limits it to first 75
   - Or file not found → uses hardcoded 75 font list

---

## Complete Fix Commands

```python
# CELL 1: Navigate and check
import os
import json
os.chdir('/content/drive/MyDrive/check_fonts')

# CELL 2: Verify JSON file exists and has correct data
if os.path.exists('downloaded_fonts_list.json'):
    with open('downloaded_fonts_list.json', 'r') as f:
        data = json.load(f)
        num_fonts = len(data.get('available', []))
        print(f"✅ Found {num_fonts} fonts in JSON file")
        
        # CELL 3: Run data_collection with correct number
        !python data_collection.py --num_fonts {num_fonts} --num_samples 20
else:
    print("❌ downloaded_fonts_list.json not found!")
    print("Make sure you ran get_all_google_fonts.py from check_fonts folder")
```

---

## Quick Diagnostic

Run this to see what's happening:

```python
import os
import json
from pathlib import Path

os.chdir('/content/drive/MyDrive/check_fonts')

# Check 1: Does JSON file exist?
json_file = Path('downloaded_fonts_list.json')
print(f"JSON file exists: {json_file.exists()}")
print(f"JSON file path: {json_file.absolute()}")

# Check 2: What's in it?
if json_file.exists():
    with open(json_file, 'r') as f:
        data = json.load(f)
        print(f"\nJSON contents:")
        print(f"  Total: {data.get('total', 0)}")
        print(f"  Available: {len(data.get('available', []))}")
        print(f"  Missing: {len(data.get('missing', []))}")
        print(f"\nFirst 10 fonts: {data.get('available', [])[:10]}")

# Check 3: What will data_collection.py see?
from data_collection import FontDatasetCollector
collector = FontDatasetCollector(num_fonts=None)  # Don't limit
print(f"\nFonts collector will use: {len(collector.target_fonts)}")
print(f"First 10: {collector.target_fonts[:10]}")
```

---

## Summary

**The issue:** `data_collection.py` defaults to 75 fonts even if 1000 are downloaded.

**Quick fix:** Run with `--num_fonts 1000`

**Proper fix:** 
1. Make sure `downloaded_fonts_list.json` is in `check_fonts/` folder
2. Run: `!python data_collection.py --num_fonts 1000 --num_samples 20`
3. Or update code to auto-detect (already done!)

**Check:** Run the diagnostic above to see what's happening! 🔍



