"""
Utility script to check available fonts on your system
Helps identify which fonts from our target list are available
"""

import os
from pathlib import Path
import platform
from data_collection import FontDatasetCollector

def find_all_system_fonts():
    """Find all fonts available on the system"""
    fonts = []
    
    if platform.system() == 'Windows':
        font_dir = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
        if font_dir.exists():
            for ext in ['*.ttf', '*.otf']:
                fonts.extend([f.stem for f in font_dir.glob(ext)])
    elif platform.system() == 'Linux':
        font_dirs = [
            '/usr/share/fonts/truetype',
            '/usr/share/fonts/opentype',
            Path.home() / '.fonts',
        ]
        for font_dir in font_dirs:
            font_dir = Path(font_dir).expanduser()
            if font_dir.exists():
                for ext in ['*.ttf', '*.otf']:
                    fonts.extend([f.stem for f in font_dir.rglob(ext)])
    elif platform.system() == 'Darwin':
        font_dirs = [
            '/Library/Fonts',
            Path.home() / 'Library/Fonts',
            '/System/Library/Fonts',
        ]
        for font_dir in font_dirs:
            font_dir = Path(font_dir).expanduser()
            if font_dir.exists():
                for ext in ['*.ttf', '*.otf']:
                    fonts.extend([f.stem for f in font_dir.rglob(ext)])
    
    return list(set(fonts))  # Remove duplicates

def check_target_fonts_availability():
    """Check which target fonts are available on the system"""
    collector = FontDatasetCollector()
    system_fonts = find_all_system_fonts()
    
    available = []
    missing = []
    
    print("Checking font availability...\n")
    print("="*60)
    
    for font_name in collector.target_fonts:
        # Check if font exists (case-insensitive)
        font_lower = font_name.lower()
        found = False
        
        for sys_font in system_fonts:
            if font_lower in sys_font.lower() or sys_font.lower() in font_lower:
                available.append(font_name)
                found = True
                break
        
        if not found:
            # Try finding via the find_font_file method
            font_file = collector.find_font_file(font_name)
            if font_file:
                available.append(font_name)
                found = True
            else:
                missing.append(font_name)
    
    print(f"\n Available fonts: {len(available)}/{len(collector.target_fonts)}")
    print(f" Missing fonts: {len(missing)}/{len(collector.target_fonts)}\n")
    
    if available:
        print("Available fonts:")
        for font in available[:20]:  # Show first 20
            print(f"  ✓ {font}")
        if len(available) > 20:
            print(f"  ... and {len(available) - 20} more")
    
    if missing:
        print(f"\nMissing fonts (first 20):")
        for font in missing[:20]:
            print(f"  ✗ {font}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    
    print("\n" + "="*60)
    print("\nRecommendations:")
    if len(available) < 30:
        print("   You have fewer than 30 fonts available.")
        print("   Consider downloading Google Fonts manually or using the API.")
    else:
        print("   You have enough fonts to start! Proceed with data collection.")
    
    return available, missing

if __name__ == "__main__":
    check_target_fonts_availability()

