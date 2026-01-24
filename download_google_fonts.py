"""
Google Fonts Downloader
Automatically downloads fonts from Google Fonts API for use in the project
"""

import os
import requests
import json
from pathlib import Path
from tqdm import tqdm
import zipfile
import shutil
import re

class GoogleFontsDownloader:
    def __init__(self, output_dir="downloaded_fonts", api_key=None):
        """
        Initialize the Google Fonts downloader
        
        Args:
            output_dir: Directory to save downloaded fonts
            api_key: Google Fonts API key (optional for public API)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.api_url = "https://www.googleapis.com/webfonts/v1/webfonts"
        
    def get_available_fonts(self):
        """Get list of all available Google Fonts"""
        try:
            params = {}
            if self.api_key:
                params['key'] = self.api_key
            
            response = requests.get(self.api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {font['family']: font for font in data.get('items', [])}
            else:
                # API request failed - this is OK, we'll use fallback method
                print(f"⚠️  API request failed with status {response.status_code}")
                print("   (This is normal - using direct download method instead)")
                if response.status_code == 403 or response.status_code == 401:
                    print("   Note: Google Fonts API may require an API key.")
                    print("   But don't worry - direct download works without one!")
                return {}
        except Exception as e:
            print(f"Error fetching Google Fonts: {e}")
            return {}
    
    def download_font(self, font_name, font_info=None):
        """
        Download a specific font from Google Fonts
        
        Args:
            font_name: Name of the font to download
            font_info: Font metadata (optional, will fetch if not provided)
        """
        # Create font directory
        font_dir = self.output_dir / font_name.replace(' ', '_')
        font_dir.mkdir(exist_ok=True)
        
        # Check if font already exists (skip if already downloaded)
        if font_dir.exists():
            existing_files = list(font_dir.glob('*.ttf')) + list(font_dir.glob('*.otf'))
            if existing_files:
                return True  # Font already downloaded, skip
        
        # If font_info not provided, try to get it
        if font_info is None:
            fonts_dict = self.get_available_fonts()
            font_info = fonts_dict.get(font_name)
        
        if not font_info:
            print(f"⚠️  Font '{font_name}' not found in Google Fonts")
            return False
        
        # Download each variant
        downloaded_files = []
        
        # Get all available files for this font
        files = font_info.get('files', {})
        
        if not files:
            print(f"⚠️  No files available for '{font_name}'")
            return False
        
        # Download regular (400) weight first, then others
        variants_to_download = ['regular', '400', '700', 'italic']
        
        for variant in variants_to_download:
            if variant in files:
                file_url = files[variant]
                try:
                    # Download the font file
                    response = requests.get(file_url, timeout=30)
                    if response.status_code == 200:
                        # Determine file extension
                        ext = '.ttf' if 'ttf' in file_url else '.woff2'
                        if ext == '.woff2':
                            # Skip woff2, prefer ttf
                            continue
                        
                        # Save file
                        filename = f"{font_name.replace(' ', '_')}_{variant}{ext}"
                        file_path = font_dir / filename
                        
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        
                        downloaded_files.append(file_path)
                except Exception as e:
                    print(f"  Error downloading {variant}: {e}")
        
        # If no files downloaded, try any available variant
        if not downloaded_files:
            for variant, file_url in files.items():
                try:
                    response = requests.get(file_url, timeout=30)
                    if response.status_code == 200:
                        ext = '.ttf' if 'ttf' in file_url else '.woff2'
                        if ext == '.woff2':
                            continue
                        
                        filename = f"{font_name.replace(' ', '_')}_{variant}{ext}"
                        file_path = font_dir / filename
                        
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        
                        downloaded_files.append(file_path)
                        break  # Just get one file
                except Exception as e:
                    continue
        
        if downloaded_files:
            print(f"✅ Downloaded {font_name}: {len(downloaded_files)} file(s)")
            return True
        else:
            print(f"❌ Failed to download {font_name}")
            return False
    
    def download_fonts_list(self, font_names):
        """
        Download multiple fonts from a list
        
        Args:
            font_names: List of font names to download
        """
        print(f"Fetching Google Fonts catalog...")
        fonts_dict = self.get_available_fonts()
        
        if not fonts_dict:
            print("⚠️  Could not fetch Google Fonts catalog")
            print("Trying CSS API download method (more reliable)...")
            # Fallback: use CSS API method which is more reliable
            available = []
            missing = []
            for font_name in tqdm(font_names, desc="Downloading fonts"):
                # Check if already downloaded first
                font_dir = self.output_dir / font_name.replace(' ', '_')
                if font_dir.exists():
                    existing_files = list(font_dir.glob('*.ttf')) + list(font_dir.glob('*.otf'))
                    if existing_files:
                        available.append(font_name)
                        continue  # Skip, already downloaded
                
                # Try CSS API method first (more reliable)
                font_slug = font_name.lower().replace(' ', '-')
                if self.download_font_via_cdn(font_name, font_slug):
                    available.append(font_name)
                elif self.download_font_direct(font_name):
                    available.append(font_name)
                else:
                    missing.append(font_name)
            
            print(f"\n{'='*60}")
            print(f"✅ Successfully downloaded: {len(available)} fonts")
            print(f"❌ Failed to download: {len(missing)} fonts")
            
            if missing:
                print(f"\n⚠️  Some fonts failed to download. This might be due to:")
                print("   - Network issues")
                print("   - Font name variations")
                print("   - Rate limiting")
                print(f"\nYou can run the script again to retry failed fonts.")
            
            return available, missing
        
        print(f"Found {len(fonts_dict)} fonts in Google Fonts catalog\n")
        
        available = []
        missing = []
        
        for font_name in tqdm(font_names, desc="Downloading fonts"):
            if font_name in fonts_dict:
                result = self.download_font(font_name, fonts_dict[font_name])
                if result:
                    available.append(font_name)
                else:
                    missing.append(font_name)
            else:
                # Try alternative name formats
                found = False
                for key in fonts_dict.keys():
                    if font_name.lower() == key.lower():
                        if self.download_font(key, fonts_dict[key]):
                            available.append(key)
                            found = True
                        break
                
                if not found:
                    missing.append(font_name)
        
        print(f"\n{'='*60}")
        print(f"✅ Successfully downloaded: {len(available)} fonts")
        print(f"❌ Failed to download: {len(missing)} fonts")
        
        if missing:
            print(f"\nMissing fonts: {', '.join(missing[:10])}")
            if len(missing) > 10:
                print(f"... and {len(missing) - 10} more")
        
        return available, missing
    
    def download_font_direct(self, font_name):
        """
        Alternative method: Download font directly from Google Fonts CDN
        Uses the actual font file URLs from Google Fonts
        """
        font_dir = self.output_dir / font_name.replace(' ', '_')
        font_dir.mkdir(exist_ok=True)
        
        # Check if font already exists (skip if already downloaded)
        if font_dir.exists():
            existing_files = list(font_dir.glob('*.ttf')) + list(font_dir.glob('*.otf'))
            if existing_files:
                return True  # Font already downloaded, skip
        
        # Google Fonts CDN base URL pattern
        # We'll try to construct the URL for the regular weight font
        font_slug = font_name.lower().replace(' ', '-')
        
        # Try multiple URL patterns that Google Fonts uses
        cdn_patterns = [
            f"https://fonts.gstatic.com/s/{font_slug}/v{version}/"
            for version in range(1, 30)  # Try different versions
        ]
        
        # More reliable: Use the Google Fonts API to get the actual file URLs
        # But if that fails, try direct CDN access
        try:
            # Try the download endpoint with proper headers
            font_url_name = font_name.replace(' ', '%20')
            download_url = f"https://fonts.google.com/download?family={font_url_name}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/zip, application/octet-stream, */*',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            session = requests.Session()
            response = session.get(download_url, headers=headers, timeout=30, allow_redirects=True)
            
            if response.status_code == 200:
                # Check content type and first bytes to see if it's a zip
                content = response.content
                content_type = response.headers.get('content-type', '').lower()
                
                # Check if it's actually a zip file (starts with PK header)
                is_zip = content[:2] == b'PK' or 'zip' in content_type or 'octet-stream' in content_type
                
                if is_zip:
                    # Save and extract zip file
                    zip_path = font_dir / f"{font_name.replace(' ', '_')}.zip"
                    with open(zip_path, 'wb') as f:
                        f.write(content)
                    
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(font_dir)
                        zip_path.unlink()  # Remove zip after extraction
                        
                        # Verify files were extracted
                        if list(font_dir.glob('*.ttf')) or list(font_dir.glob('*.otf')):
                            print(f"✅ Downloaded {font_name} via direct method")
                            return True
                        else:
                            zip_path.unlink() if zip_path.exists() else None
                            return False
                    except zipfile.BadZipFile:
                        zip_path.unlink() if zip_path.exists() else None
                        return False
                else:
                    # Not a zip - might be HTML page, try parsing it or use alternative method
                    return self.download_font_via_cdn(font_name, font_slug)
            else:
                # Try CDN method as fallback
                return self.download_font_via_cdn(font_name, font_slug)
                
        except Exception as e:
            # Try CDN method as fallback
            return self.download_font_via_cdn(font_name, font_slug)
    
    def download_font_via_cdn(self, font_name, font_slug):
        """
        Try to download font files directly from Google Fonts CDN
        This is a fallback method when the download endpoint doesn't work
        """
        font_dir = self.output_dir / font_name.replace(' ', '_')
        
        # Check if font already exists (skip if already downloaded)
        if font_dir.exists():
            existing_files = list(font_dir.glob('*.ttf')) + list(font_dir.glob('*.otf'))
            if existing_files:
                return True  # Font already downloaded, skip
        
        # Use gstatic.com CDN - try common font file names
        # Note: This is less reliable but sometimes works
        cdn_base = "https://fonts.gstatic.com/s"
        
        # Common font file naming patterns
        file_patterns = [
            f"{font_slug}/v*/{font_slug}-regular.ttf",
            f"{font_slug}/v*/{font_slug.replace('-', '')}-regular.ttf",
        ]
        
        # For MVP, let's use a Python package that handles this better
        # But first, let's try one more thing - use requests-html or selenium
        # Actually, simpler: use the fonts.googleapis.com CSS and parse it
        
        try:
            # Get the CSS file which contains font URLs
            # This is the most reliable method - Google Fonts CSS API
            css_url = f"https://fonts.googleapis.com/css2?family={font_name.replace(' ', '+')}:wght@400"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/css,*/*;q=0.1'
            }
            response = requests.get(css_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                css_content = response.text
                # Parse CSS to find font URLs (look for .ttf files specifically)
                font_urls = re.findall(r'url\((https://fonts\.gstatic\.com/[^)]+\.ttf[^)]*)\)', css_content)
                
                # If no .ttf found, try any font file
                if not font_urls:
                    font_urls = re.findall(r'url\((https://fonts\.gstatic\.com/[^)]+)\)', css_content)
                
                if font_urls:
                    # Download the first font file found
                    font_url = font_urls[0]
                    font_response = requests.get(font_url, timeout=30)
                    
                    if font_response.status_code == 200:
                        # Determine file extension from URL
                        ext = '.ttf' if '.ttf' in font_url else '.otf' if '.otf' in font_url else '.woff2'
                        if ext == '.woff2':
                            return False  # Skip woff2
                        
                        filename = f"{font_name.replace(' ', '_')}_regular{ext}"
                        file_path = font_dir / filename
                        
                        with open(file_path, 'wb') as f:
                            f.write(font_response.content)
                        
                        print(f"✅ Downloaded {font_name} via CDN")
                        return True
        except Exception as e:
            pass
        
        return False

def main():
    """Main execution"""
    from data_collection import FontDatasetCollector
    
    # Get target fonts from the collector
    collector = FontDatasetCollector()
    target_fonts = collector.target_fonts
    
    print("="*60)
    print("Google Fonts Downloader")
    print("="*60)
    print(f"\nTarget: Download {len(target_fonts)} fonts")
    print(f"Output directory: downloaded_fonts/\n")
    
    downloader = GoogleFontsDownloader(output_dir="downloaded_fonts")
    
    # Download fonts
    available, missing = downloader.download_fonts_list(target_fonts)
    
    print(f"\n{'='*60}")
    print("Download Complete!")
    print("="*60)
    print(f"\nFonts saved to: {downloader.output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Update data_collection.py to use downloaded fonts")
    print("2. Run: python data_collection.py")

if __name__ == "__main__":
    main()

