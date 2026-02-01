# Understanding System Fonts

## What are System Fonts?

**System fonts** are fonts that come pre-installed with your operating system (Windows, macOS, or Linux). They're stored in specific directories on your computer and are available to all applications.

## What Does "Scanning System Fonts" Mean?

When `check_fonts.py` scans system fonts, it:

1. **Looks in specific directories** where your OS stores font files
2. **Finds all `.ttf` and `.otf` files** (TrueType and OpenType font formats)
3. **Creates a list** of all available font names
4. **Compares** that list against the 75 target fonts we want for the dataset

## Where Are System Fonts Stored?

### On Windows (Your System):
```
C:\Windows\Fonts\
```
This folder contains all fonts installed on your Windows system. Examples:
- **Arial** - Common sans-serif font
- **Times New Roman** - Classic serif font
- **Calibri** - Default Office font
- **Segoe UI** - Windows UI font
- **Courier New** - Monospace font

### On Linux:
```
/usr/share/fonts/truetype/
/usr/share/fonts/opentype/
~/.fonts/  (user-specific fonts)
```

### On macOS:
```
/Library/Fonts/
~/Library/Fonts/
/System/Library/Fonts/
```

## Common Windows System Fonts

When you run `check_fonts.py` on Windows, you'll typically find fonts like:

**Sans-serif fonts:**
- Arial
- Calibri
- Segoe UI
- Verdana
- Tahoma

**Serif fonts:**
- Times New Roman
- Georgia
- Cambria

**Monospace fonts:**
- Courier New
- Consolas

**Decorative/Display:**
- Comic Sans MS
- Impact
- Arial Black

## Why Does the Script Scan System Fonts?

The script needs to know:
1. **Which fonts you already have** - so it can use them directly
2. **Which fonts are missing** - so it knows what needs to be downloaded or uses fallback

## Example: What Happens During Scanning

```
1. Script checks: C:\Windows\Fonts\
2. Finds files like: arial.ttf, times.ttf, calibri.ttf, etc.
3. Extracts font names: "Arial", "Times New Roman", "Calibri"
4. Compares with target list: ["Roboto", "Open Sans", "Arial", ...]
5. Reports: "Found Arial ✓, Missing Roboto ✗"
```

## What If Fonts Are Missing?

If the script finds that many target fonts are missing:
- The `data_collection.py` script will still work
- It will use **default fonts** as fallback for missing ones
- For best results, you'd want to install Google Fonts manually or via API

## How to See Your System Fonts Manually

**On Windows:**
1. Open File Explorer
2. Navigate to `C:\Windows\Fonts`
3. You'll see all installed fonts

**Or via PowerShell:**
```powershell
Get-ChildItem C:\Windows\Fonts\*.ttf | Select-Object Name
```

## Summary

- **System fonts** = Fonts that come with your OS
- **Scanning** = Looking through font directories to find what's available
- **Purpose** = Know what fonts we can use vs. what we need to download

The scanning process is just the script being thorough - it wants to make sure it uses real fonts when possible, rather than falling back to defaults!

