"""
Quick utility to check if fonts were downloaded and where they are
"""

from pathlib import Path

def check_downloaded_fonts():
    """Check if downloaded_fonts directory exists and what's in it"""
    script_dir = Path(__file__).parent
    downloaded_fonts_dir = script_dir / "downloaded_fonts"
    
    print("="*60)
    print("Checking Downloaded Fonts")
    print("="*60)
    print(f"\nLooking in: {downloaded_fonts_dir.absolute()}")
    
    if not downloaded_fonts_dir.exists():
        print("\n❌ 'downloaded_fonts' directory does NOT exist!")
        print("\nYou need to run: python download_google_fonts.py")
        print("This will download all the fonts you need.")
        return False
    
    # Count fonts
    font_dirs = [d for d in downloaded_fonts_dir.iterdir() if d.is_dir()]
    font_files = []
    
    for font_dir in font_dirs:
        ttf_files = list(font_dir.glob('*.ttf'))
        otf_files = list(font_dir.glob('*.otf'))
        if ttf_files or otf_files:
            font_files.append((font_dir.name, len(ttf_files) + len(otf_files)))
    
    print(f"\n✅ Found 'downloaded_fonts' directory")
    print(f"   Font folders: {len(font_dirs)}")
    print(f"   Fonts with files: {len(font_files)}")
    
    if font_files:
        print(f"\nSample fonts found:")
        for font_name, file_count in font_files[:10]:
            print(f"  ✓ {font_name} ({file_count} file(s))")
        if len(font_files) > 10:
            print(f"  ... and {len(font_files) - 10} more")
    else:
        print("\n⚠️  No font files (.ttf or .otf) found in font directories")
        print("   The directories exist but are empty.")
        print("   Try running: python download_google_fonts.py")
    
    return len(font_files) > 0

if __name__ == "__main__":
    check_downloaded_fonts()




