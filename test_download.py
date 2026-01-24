"""
Test script to debug font downloading
Tests downloading a single font to see what's happening
"""

from download_google_fonts import GoogleFontsDownloader

def test_single_font():
    """Test downloading a single font"""
    downloader = GoogleFontsDownloader(output_dir="test_fonts")
    
    # Test with a common font
    test_font = "Roboto"
    
    print(f"Testing download of: {test_font}")
    print("="*60)
    
    # Try direct download method
    result = downloader.download_font_direct(test_font)
    
    if result:
        print(f"\n✅ Successfully downloaded {test_font}")
        
        # Check what files were created
        font_dir = downloader.output_dir / test_font.replace(' ', '_')
        files = list(font_dir.glob('*'))
        print(f"\nFiles created:")
        for f in files:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
    else:
        print(f"\n❌ Failed to download {test_font}")
        print("\nTrying alternative method...")
        
        # Try via CSS API
        import requests
        import re
        
        css_url = f"https://fonts.googleapis.com/css2?family={test_font.replace(' ', '+')}:wght@400"
        response = requests.get(css_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            print(f"CSS API response received ({len(response.text)} chars)")
            font_urls = re.findall(r'url\((https://fonts\.gstatic\.com/[^)]+\.ttf[^)]*)\)', response.text)
            print(f"Found {len(font_urls)} font URLs in CSS")
            if font_urls:
                print(f"First URL: {font_urls[0]}")

if __name__ == "__main__":
    test_single_font()




