# Troubleshooting: Fonts Not Found

## Problem: "Font not found, using default font"

This happens when `data_collection.py` can't find the fonts. Here's how to fix it:

## Solution: Download Fonts First!

You **must** run the download script **before** running data collection:

### Step 1: Download Google Fonts

```powershell
python download_google_fonts.py
```

**Wait for this to complete!** It will:
- Download 75 fonts from Google Fonts
- Save them to `downloaded_fonts/` folder
- Show progress for each font
- Take 5-10 minutes

### Step 2: Verify Fonts Were Downloaded

Check that fonts exist:

```powershell
python check_downloaded_fonts.py
```

Or manually check:

```powershell
dir downloaded_fonts
```

You should see folders like:
- `Roboto/`
- `Open_Sans/`
- `Montserrat/`
- etc.

### Step 3: Run Data Collection

**Only after fonts are downloaded:**

```powershell
python data_collection.py
```

## Why This Happens

The `data_collection.py` script looks for fonts in this order:

1. ✅ `downloaded_fonts/` folder (Google Fonts)
2. ✅ System fonts (`C:\Windows\Fonts\`)
3. ❌ Default font (fallback)

If step 1 fails (no downloaded fonts), it tries step 2 (system fonts). Since you only have Courier Prime, it falls back to default font.

## Quick Check Script

Run this to see what's happening:

```powershell
python check_downloaded_fonts.py
```

This will tell you:
- ✅ If fonts are downloaded
- ❌ If you need to download them
- 📊 How many fonts were found

## Common Issues

### Issue: "downloaded_fonts directory does not exist"
**Solution**: Run `python download_google_fonts.py` first

### Issue: "Font folders exist but are empty"
**Solution**: The download may have failed. Check your internet connection and run the download script again.

### Issue: "Some fonts downloaded but not all"
**Solution**: This is OK! The script will use what's available. You can run the download script again to retry failed fonts.

## Correct Order of Operations

```
1. python download_google_fonts.py  ← Download fonts FIRST
2. python check_downloaded_fonts.py  ← Verify (optional)
3. python data_collection.py         ← Generate samples
```

**Don't skip step 1!** 🚨

