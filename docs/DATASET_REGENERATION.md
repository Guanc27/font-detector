# Dataset Regeneration Guide

## When to Re-run Data Collection

You should re-run `data_collection.py` when you want to:

1. **Change number of samples per font** (e.g., from 10 to 20)
2. **Regenerate samples** (if fonts were updated)
3. **Add more fonts** to the dataset
4. **Fix issues** with existing samples

## What Happens When You Re-run

### File Overwriting

When you re-run with `num_samples=20`:

- **Old samples** (`sample_000.png` to `sample_009.png`) will be **overwritten**
- **New samples** (`sample_000.png` to `sample_019.png`) will be created
- **Metadata** (`metadata.json`) will be **replaced** with new data

### Example:

**Before** (10 samples):
```
font_dataset/samples/Roboto/
  ├── sample_000.png
  ├── sample_001.png
  ...
  └── sample_009.png
```

**After** (20 samples):
```
font_dataset/samples/Roboto/
  ├── sample_000.png  (overwritten)
  ├── sample_001.png  (overwritten)
  ...
  ├── sample_009.png  (overwritten)
  ├── sample_010.png  (new)
  ├── sample_011.png  (new)
  ...
  └── sample_019.png  (new)
```

## Steps to Regenerate

### Option 1: Quick Regeneration (Recommended)

1. **Change the parameter** in `data_collection.py`:
   ```python
   # Line 310, change from:
   samples = self.create_font_samples(font_name, num_samples=10)
   # To:
   samples = self.create_font_samples(font_name, num_samples=20)
   ```

2. **Re-run the script**:
   ```powershell
   python data_collection.py
   ```

3. **Done!** Old samples will be overwritten, new ones added.

### Option 2: Clean Regeneration (If You Want Fresh Start)

1. **Delete old dataset** (optional):
   ```powershell
   Remove-Item -Recurse -Force font_dataset\samples
   Remove-Item font_dataset\metadata.json
   ```

2. **Run data collection**:
   ```powershell
   python data_collection.py
   ```

## Important Notes

### ⚠️ Overwriting is Fine

- Samples are **randomly generated** each time
- Old samples aren't "special" - they're just random combinations
- Overwriting is safe and expected

### ✅ Metadata Gets Updated

- `metadata.json` will reflect the new number of samples
- Training script will automatically use the new dataset
- No manual updates needed

### ⏱️ Time Considerations

- **10 samples**: ~5-10 minutes for 75 fonts
- **20 samples**: ~10-20 minutes for 75 fonts
- **30 samples**: ~15-30 minutes for 75 fonts

## After Regeneration

1. **Verify samples**:
   ```powershell
   # Check sample count
   Get-ChildItem font_dataset\samples\*\*.png | Measure-Object | Select-Object Count
   ```

2. **Check metadata**:
   ```powershell
   python -c "import json; d=json.load(open('font_dataset/metadata.json')); print(f'Samples: {sum(f[\"num_samples\"] for f in d[\"fonts\"])}')"
   ```

3. **Re-train model** (if you already trained):
   - Old model was trained on 10 samples per font
   - New model should be trained on 20 samples per font
   - Better accuracy expected!

## Changing num_samples Parameter

### In `create_font_samples` method (Line 247):

This is the **default parameter** - only used if not specified:

```python
def create_font_samples(self, font_name, num_samples=20):
```

### In `collect_dataset` method (Line 310):

This is where you **actually specify** how many samples:

```python
samples = self.create_font_samples(font_name, num_samples=20)
```

**Change this line** to control the number of samples!

## Recommendations

### For MVP (Quick Start):
- **10 samples per font** = Fast, good enough for testing
- **Total**: ~750 samples

### For Better Accuracy:
- **20 samples per font** = Better diversity, improved accuracy
- **Total**: ~1,500 samples

### For Best Results:
- **30 samples per font** = Maximum diversity, best accuracy
- **Total**: ~2,250 samples

## FAQ

**Q: Will I lose my old samples?**  
A: Yes, but that's fine - they're randomly generated anyway.

**Q: Do I need to delete the old dataset?**  
A: No, the script will overwrite automatically.

**Q: Will training work with the new dataset?**  
A: Yes, but you should re-train the model for best results.

**Q: How long does regeneration take?**  
A: About 1-2 minutes per font, so ~75-150 minutes for 75 fonts.

**Q: Can I keep both old and new samples?**  
A: Not easily - you'd need to modify the script to use different filenames.




