# Quick Start Guide - Downloading Google Fonts

Since you only have Courier Prime available, let's download Google Fonts automatically!

## Step 1: Download Google Fonts

Run the downloader script:

```powershell
python download_google_fonts.py
```

This will:
- Connect to Google Fonts API
- Download all 75 target fonts
- Save them to `downloaded_fonts/` directory
- Show progress for each font

**Expected time**: 5-10 minutes (depending on internet speed)

## Step 2: Verify Downloads

Check that fonts were downloaded:

```powershell
dir downloaded_fonts
```

You should see folders for each font (e.g., `Roboto`, `Open_Sans`, etc.)

## Step 3: Run Data Collection

Now run the data collection script - it will automatically use the downloaded fonts:

```powershell
python data_collection.py
```

## How It Works

1. **`download_google_fonts.py`** downloads fonts from Google Fonts API
2. Fonts are saved to `downloaded_fonts/` folder
3. **`data_collection.py`** automatically checks this folder first before looking in system fonts
4. Font samples are generated using the downloaded fonts

## Troubleshooting

### If download fails:
- Check your internet connection
- Google Fonts API is free but may have rate limits
- Try running again - it will skip already downloaded fonts

### If some fonts fail to download:
- The script will continue with available fonts
- You can run it again to retry failed fonts
- Check the output for which fonts succeeded/failed

### Alternative: Manual Download
If automatic download doesn't work, you can manually download fonts from:
https://fonts.google.com/

Then place them in the `downloaded_fonts/` folder.

## Next Steps

After fonts are downloaded and samples are generated:
- Proceed to Phase 2: Model Training
- The dataset will be ready in `font_dataset/` folder

