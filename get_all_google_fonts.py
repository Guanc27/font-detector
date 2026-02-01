"""
Get all available Google Fonts and download them
Fetches complete list from Google Fonts API and downloads all fonts
"""

from download_google_fonts import GoogleFontsDownloader
import requests
import json

def get_all_google_fonts():
    """Fetch all available fonts from Google Fonts"""
    try:
        # Method 1: Try Google Fonts API
        url = "https://www.googleapis.com/webfonts/v1/webfonts"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                fonts_data = response.json()
                all_fonts = [font['family'] for font in fonts_data.get('items', [])]
                print(f"✅ Fetched {len(all_fonts)} fonts from Google Fonts API")
                return all_fonts
            else:
                print(f"⚠️  API returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  API request failed: {e}")
        
        # Method 2: Use Google Fonts metadata endpoint (most reliable)
        print("Trying Google Fonts metadata endpoint...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        try:
            # This endpoint returns all fonts in JSON format
            metadata_url = "https://fonts.google.com/metadata/fonts"
            response = requests.get(metadata_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # The response is a JSON array of font objects
                fonts_data = response.json()
                
                # Debug: Print first few items to understand structure
                if isinstance(fonts_data, list) and len(fonts_data) > 0:
                    print(f"   Sample font data: {fonts_data[0]}")
                elif isinstance(fonts_data, dict):
                    print(f"   Response keys: {list(fonts_data.keys())[:5]}")
                
                # Handle different response formats
                if isinstance(fonts_data, list):
                    # Direct array of font objects - try different field names
                    all_fonts = []
                    for font in fonts_data:
                        # Try common field names
                        font_name = font.get('family') or font.get('name') or font.get('familyName') or font.get('fontFamily')
                        if font_name:
                            all_fonts.append(font_name)
                elif isinstance(fonts_data, dict):
                    # Dictionary with various possible keys
                    all_fonts = []
                    for key in ['familyMetadataList', 'fonts', 'items', 'families', 'fontList']:
                        if key in fonts_data and isinstance(fonts_data[key], list):
                            for item in fonts_data[key]:
                                font_name = item.get('family') or item.get('name') or item.get('familyName') or item.get('fontFamily')
                                if font_name:
                                    all_fonts.append(font_name)
                            if all_fonts:
                                break
                else:
                    all_fonts = []
                
                if all_fonts:
                    # Remove empty strings and duplicates
                    all_fonts = [f for f in all_fonts if f]
                    all_fonts = list(dict.fromkeys(all_fonts))
                    print(f"✅ Found {len(all_fonts)} fonts from metadata endpoint")
                    return all_fonts
                else:
                    print(f"⚠️  Metadata endpoint returned data but couldn't parse font names")
                    print(f"   Response type: {type(fonts_data)}")
                    if isinstance(fonts_data, dict):
                        print(f"   Keys: {list(fonts_data.keys())[:10]}")
                    elif isinstance(fonts_data, list) and len(fonts_data) > 0:
                        print(f"   First item keys: {list(fonts_data[0].keys())[:10] if isinstance(fonts_data[0], dict) else 'Not a dict'}")
        except Exception as e:
            print(f"⚠️  Metadata endpoint failed: {e}")
        
        # Method 3: Try alternative API endpoints
        print("Trying alternative API endpoints...")
        try:
            # Try the webfonts API with different parameters
            alt_urls = [
                "https://www.googleapis.com/webfonts/v1/webfonts?sort=popularity",
                "https://www.googleapis.com/webfonts/v1/webfonts?sort=alpha",
            ]
            
            for alt_url in alt_urls:
                try:
                    response = requests.get(alt_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        fonts_data = response.json()
                        all_fonts = [font['family'] for font in fonts_data.get('items', [])]
                        if all_fonts:
                            print(f"✅ Found {len(all_fonts)} fonts from alternative API")
                            return all_fonts
                except:
                    continue
        except Exception as e:
            print(f"⚠️  Alternative endpoints failed: {e}")
        
        # Method 4: Use extended hardcoded list as fallback
        print("⚠️  All API methods failed, using extended font list...")
        
        # Extended list of popular Google Fonts
        extended_fonts = [
            # Original 75 fonts
            "Roboto", "Open Sans", "Lato", "Montserrat", "Source Sans Pro",
            "Raleway", "PT Sans", "Ubuntu", "Noto Sans", "Oswald",
            "Roboto Condensed", "Slabo 27px", "Roboto Slab", "Merriweather",
            "PT Serif", "Lora", "Crimson Text", "Playfair Display", "Bitter",
            "Poppins", "Nunito", "Inter", "Work Sans", "Fira Sans",
            "Rubik", "Quicksand", "Dosis", "Exo 2", "Comfortaa",
            "Bebas Neue", "Anton", "Righteous", "Lobster", "Pacifico",
            "Dancing Script", "Satisfy", "Great Vibes", "Amatic SC",
            "Roboto Mono", "Source Code Pro", "Fira Code", "Courier Prime",
            "Playfair Display SC", "Cinzel", "Abril Fatface", "Bangers",
            "Crete Round", "Droid Sans", "Droid Serif", "Indie Flower",
            "Josefin Sans", "Josefin Slab", "Karla", "Libre Baskerville",
            "Libre Franklin", "Muli", "Nunito Sans", "Old Standard TT",
            "Oxygen", "PT Sans Narrow", "Quattrocento", "Quattrocento Sans",
            "Rajdhani", "Roboto Mono", "Sarabun", "Titillium Web",
            "Varela Round", "Yanone Kaffeesatz", "Zilla Slab",
            
            # Additional popular fonts (100+ more)
            "Alegreya", "Alegreya Sans", "Archivo", "Archivo Narrow",
            "Arimo", "Arvo", "Asap", "Barlow", "Barlow Condensed",
            "Barlow Semi Condensed", "Bitter", "Cabin", "Cabin Condensed",
            "Cantarell", "Cardo", "Chivo", "Comfortaa", "Cormorant",
            "Cormorant Garamond", "Crimson Pro", "DM Sans", "DM Serif Display",
            "EB Garamond", "Encode Sans", "Encode Sans Condensed",
            "Encode Sans Expanded", "Encode Sans Semi Condensed",
            "Fira Mono", "Fira Sans Condensed", "Fraunces", "IBM Plex Mono",
            "IBM Plex Sans", "IBM Plex Serif", "Inconsolata", "Inria Sans",
            "Inria Serif", "Josefin Sans", "Josefin Slab", "Kumbh Sans",
            "Lexend", "Libre Caslon Text", "Manrope", "Martel", "Merriweather Sans",
            "Mukta", "Mukta Mahee", "Mukta Malar", "Mukta Vaani",
            "Nanum Gothic", "Nanum Myeongjo", "Noto Serif", "Nunito",
            "Orbitron", "Overpass", "Overpass Mono", "Permanent Marker",
            "Piazzolla", "Poppins", "Prata", "Public Sans", "Quicksand",
            "Raleway", "Red Hat Display", "Red Hat Mono", "Red Hat Text",
            "Roboto Flex", "Roboto Mono", "Roboto Serif", "Rock Salt",
            "Rubik", "Rubik Mono One", "Saira", "Saira Condensed",
            "Saira Extra Condensed", "Saira Semi Condensed", "Sansita",
            "Sansita Swashed", "Sora", "Space Grotesk", "Space Mono",
            "Spectral", "Syne", "Teko", "Titillium Web", "Trirong",
            "Truculenta", "Ubuntu Condensed", "Ubuntu Mono", "Unbounded",
            "Vollkorn", "Yeseva One", "Zilla Slab Highlight"
        ]
        
        # Remove duplicates
        extended_fonts = list(dict.fromkeys(extended_fonts))
        print(f"✅ Using extended list: {len(extended_fonts)} fonts")
        return extended_fonts
        
    except Exception as e:
        print(f"Error fetching fonts: {e}")
        return []

def download_all_fonts(num_fonts=1000):
    """
    Download all available Google Fonts
    
    Args:
        num_fonts: Limit to N fonts (None = download all)
    """
    print("="*60)
    print("Downloading All Google Fonts")
    print("="*60)
    
    # Get all fonts
    all_fonts = get_all_google_fonts()
    
    if num_fonts:
        all_fonts = all_fonts[:num_fonts]
        print(f"\nLimiting to first {num_fonts} fonts")
    
    print(f"\nTotal fonts to download: {len(all_fonts)}")
    
    # Download fonts
    downloader = GoogleFontsDownloader(output_dir="downloaded_fonts")
    available, missing = downloader.download_fonts_list(all_fonts)
    
    print(f"\n{'='*60}")
    print("Download Complete!")
    print("="*60)
    print(f"✅ Successfully downloaded: {len(available)} fonts")
    print(f"❌ Failed: {len(missing)} fonts")
    
    # Save list of available fonts
    with open("downloaded_fonts_list.json", "w") as f:
        json.dump({
            "available": available,
            "missing": missing,
            "total": len(available)
        }, f, indent=2)
    
    print(f"\nFont list saved to: downloaded_fonts_list.json")
    return available, missing

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download all Google Fonts")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit to N fonts (default: download all)")
    parser.add_argument("--start_from", type=int, default=0,
                       help="Start from font N (for resuming)")
    
    args = parser.parse_args()
    
    all_fonts = get_all_google_fonts()
    
    # Determine how many fonts to download
    if args.limit:
        # User specified a limit
        limit = args.limit
        all_fonts = all_fonts[args.start_from:args.start_from + limit]
        print(f"Downloading fonts {args.start_from} to {args.start_from + limit}")
        download_all_fonts(num_fonts=len(all_fonts))  # Use sliced length
    else:
        # Use default limit of 1000 (or all if less than 1000)
        limit = min(1000, len(all_fonts))
        if args.start_from > 0:
            all_fonts = all_fonts[args.start_from:args.start_from + limit]
            print(f"Downloading fonts {args.start_from} to {args.start_from + limit} (default limit: 1000)")
        else:
            all_fonts = all_fonts[:limit]
            print(f"Downloading first {limit} fonts (default limit)")
        download_all_fonts(num_fonts=len(all_fonts))  # Use sliced length

