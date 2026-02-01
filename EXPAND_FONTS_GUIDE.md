# Guide: Downloading More Fonts for Training

## Quick Start

### Option 1: Download All Available Google Fonts

```python
# In Colab or locally
python get_all_google_fonts.py
```

This will:
- Fetch all available Google Fonts (~1000+ fonts)
- Download them to `downloaded_fonts/`
- Save list to `downloaded_fonts_list.json`

### Option 2: Download Specific Number

```python
# Download first 200 fonts
python get_all_google_fonts.py --limit 200

# Download fonts 100-200 (resume from 100)
python get_all_google_fonts.py --limit 100 --start_from 100
```

### Option 3: Manually Add Fonts

Edit `data_collection.py` and add fonts to the `target_fonts` list.

---

## Steps to Expand Your Dataset

### Step 1: Download More Fonts

```python
# Download all fonts (or limit to what you want)
python get_all_google_fonts.py --limit 200
```

**Expected time:** ~30-60 minutes for 200 fonts

### Step 2: Regenerate Dataset with More Fonts

```python
# Update data_collection.py to use more fonts
collector = FontDatasetCollector(
    output_dir="font_dataset",
    num_fonts=200  # Increase from 75
)

# Or use all downloaded fonts
collector = FontDatasetCollector(
    output_dir="font_dataset",
    num_fonts=None  # Use all available
)
```

### Step 3: Generate Samples

```python
python data_collection.py
```

This will automatically use fonts from `downloaded_fonts_list.json` if available.

---

## How Many Fonts Should You Use?

### Recommendations:

| Fonts | Samples | Training Time | Accuracy | Best For |
|-------|---------|---------------|----------|----------|
| **75** | ~1,500 | 30-60 min | 60-70% | MVP |
| **150** | ~3,000 | 1-2 hours | 70-80% | Better MVP |
| **300** | ~6,000 | 2-4 hours | 75-85% | Production |
| **500+** | ~10,000+ | 4-8 hours | 80-90% | Best accuracy |

### Trade-offs:

**More fonts:**
- ✅ Better accuracy
- ✅ More diverse dataset
- ❌ Longer training time
- ❌ More storage needed

**Fewer fonts:**
- ✅ Faster training
- ✅ Less storage
- ❌ Lower accuracy
- ❌ Less diversity

---

## Updating Your Training

After downloading more fonts:

1. **Regenerate dataset:**
   ```python
   python data_collection.py  # Will use downloaded fonts automatically
   ```

2. **Retrain model:**
   ```python
   python train_embedding_model.py --epochs 20 --batch_size 64
   ```

3. **Model will automatically adapt** to new number of fonts (classifier resizes)

---

## Tips

- **Start with 150-200 fonts** for good balance
- **Download in batches** if you want to test incrementally
- **Check downloaded fonts** before regenerating dataset
- **More fonts = better accuracy** but diminishing returns after ~300

---

## Troubleshooting

**If download fails for some fonts:**
- That's OK! The script continues with available fonts
- Check `downloaded_fonts_list.json` to see what succeeded

**If you want specific fonts:**
- Edit `get_all_google_fonts.py` to prioritize certain fonts
- Or manually add to `data_collection.py` target_fonts list

