"""
Phase 1: Font Dataset Collection Script
Downloads fonts from Google Fonts and generates training samples
"""

import os
import json
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import random

class FontDatasetCollector:
    def __init__(self, output_dir="font_dataset", num_fonts=75, num_samples=20):
        """
        Initialize the font dataset collector
        
        Args:
            output_dir: Directory to save the dataset
            num_fonts: Number of fonts to collect (default 75 for MVP, None = use all)
            num_samples: Number of samples per font (default 20)
        """
        self.output_dir = Path(output_dir)
        self.num_fonts = num_fonts
        self._num_samples = num_samples
        self.fonts_dir = self.output_dir / "fonts"
        self.samples_dir = self.output_dir / "samples"
        self.metadata_file = self.output_dir / "metadata.json"
        
        # Create directories
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Common fonts list (mix of popular and diverse styles)
        self.target_fonts = [
            # Serif fonts
            "Roboto", "Open Sans", "Lato", "Montserrat", "Source Sans Pro",
            "Raleway", "PT Sans", "Ubuntu", "Noto Sans", "Oswald",
            "Roboto Condensed", "Slabo 27px", "Roboto Slab", "Merriweather",
            "PT Serif", "Lora", "Crimson Text", "Playfair Display", "Bitter",
            
            # Sans-serif fonts
            "Poppins", "Nunito", "Inter", "Work Sans", "Fira Sans",
            "Rubik", "Quicksand", "Dosis", "Exo 2", "Comfortaa",
            
            # Display/Decorative
            "Bebas Neue", "Anton", "Righteous", "Lobster", "Pacifico",
            "Dancing Script", "Satisfy", "Great Vibes", "Amatic SC",
            
            # Monospace
            "Roboto Mono", "Source Code Pro", "Fira Code", "Courier Prime",
            
            # Additional popular fonts
            "Playfair Display SC", "Cinzel", "Abril Fatface", "Bangers",
            "Crete Round", "Droid Sans", "Droid Serif", "Indie Flower",
            "Josefin Sans", "Josefin Slab", "Karla", "Libre Baskerville",
            "Libre Franklin", "Muli", "Nunito Sans", "Old Standard TT",
            "Oxygen", "PT Sans Narrow", "Quattrocento", "Quattrocento Sans",
            "Rajdhani", "Roboto Mono", "Sarabun", "Titillium Web",
            "Varela Round", "Yanone Kaffeesatz", "Zilla Slab"
        ]
        
        # Option to load from downloaded fonts list (if available)
        downloaded_fonts_file = Path("downloaded_fonts_list.json")
        if downloaded_fonts_file.exists():
            try:
                with open(downloaded_fonts_file, 'r') as f:
                    downloaded_data = json.load(f)
                    downloaded_fonts = downloaded_data.get('available', [])
                    if downloaded_fonts:
                        print(f"Found {len(downloaded_fonts)} downloaded fonts, using those instead")
                        self.target_fonts = downloaded_fonts
            except Exception as e:
                print(f"Could not load downloaded fonts list: {e}")
        
        # Limit to num_fonts (if specified)
        if num_fonts and num_fonts < len(self.target_fonts):
            self.target_fonts = self.target_fonts[:num_fonts]
        
    def get_google_fonts_list(self):
        """Fetch available fonts from Google Fonts API"""
        try:
            url = "https://www.googleapis.com/webfonts/v1/webfonts"
            # Note: API key not strictly needed for public fonts list
            response = requests.get(url, params={"key": "AIzaSyDummy"})
            if response.status_code == 200:
                fonts_data = response.json()
                return [font['family'] for font in fonts_data.get('items', [])]
            else:
                print(f"API request failed, using predefined list. Status: {response.status_code}")
                return self.target_fonts
        except Exception as e:
            print(f"Error fetching Google Fonts list: {e}")
            print("Using predefined font list...")
            return self.target_fonts
    
    def download_font(self, font_name):
        """
        Download a font from Google Fonts
        Note: For MVP, we'll use system fonts or download manually
        This is a placeholder - actual implementation depends on your setup
        """
        font_path = self.fonts_dir / f"{font_name.replace(' ', '_')}"
        font_path.mkdir(exist_ok=True)
        
        # For MVP, we'll use system fonts if available
        # In production, you'd download from Google Fonts API
        return font_path
    
    def find_font_file(self, font_name):
        """
        Find font file on the system
        
        Args:
            font_name: Name of the font to find
            
        Returns:
            Path to font file or None if not found
        """
        import platform
        
        # First, check downloaded_fonts directory (Google Fonts)
        # Use absolute path relative to script location
        script_dir = Path(__file__).parent
        downloaded_fonts_dir = script_dir / "downloaded_fonts" / font_name.replace(' ', '_')
        
        if downloaded_fonts_dir.exists():
            # Look for .ttf files in the downloaded font directory
            ttf_files = list(downloaded_fonts_dir.glob('*.ttf'))
            if ttf_files:
                return str(ttf_files[0])  # Return first .ttf file found
            
            otf_files = list(downloaded_fonts_dir.glob('*.otf'))
            if otf_files:
                return str(otf_files[0])  # Return first .otf file found
            
            # Also check for files in subdirectories (in case zip extracted to subfolder)
            for font_file in downloaded_fonts_dir.rglob('*.ttf'):
                return str(font_file)
            for font_file in downloaded_fonts_dir.rglob('*.otf'):
                return str(font_file)
        
        # Windows font paths
        if platform.system() == 'Windows':
            windows_font_dir = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
            
            # Common variations to try
            variations = [
                font_name,
                font_name.replace(' ', ''),
                font_name.replace(' ', '-'),
                font_name + '.ttf',
                font_name + '.otf',
            ]
            
            for variation in variations:
                for ext in ['.ttf', '.otf']:
                    font_path = windows_font_dir / f"{variation}{ext}"
                    if font_path.exists():
                        return str(font_path)
            
            # Try listing all fonts and matching
            try:
                for font_file in windows_font_dir.glob('*.ttf'):
                    if font_name.lower() in font_file.stem.lower():
                        return str(font_file)
                for font_file in windows_font_dir.glob('*.otf'):
                    if font_name.lower() in font_file.stem.lower():
                        return str(font_file)
            except:
                pass
        
        # Linux font paths
        elif platform.system() == 'Linux':
            font_dirs = [
                '/usr/share/fonts/truetype',
                '/usr/share/fonts/opentype',
                '~/.fonts',
            ]
            for font_dir in font_dirs:
                font_dir = Path(font_dir).expanduser()
                if font_dir.exists():
                    for ext in ['*.ttf', '*.otf']:
                        for font_file in font_dir.rglob(ext):
                            if font_name.lower() in font_file.stem.lower():
                                return str(font_file)
        
        # macOS font paths
        elif platform.system() == 'Darwin':
            font_dirs = [
                '/Library/Fonts',
                '~/Library/Fonts',
                '/System/Library/Fonts',
            ]
            for font_dir in font_dirs:
                font_dir = Path(font_dir).expanduser()
                if font_dir.exists():
                    for ext in ['*.ttf', '*.otf']:
                        for font_file in font_dir.rglob(ext):
                            if font_name.lower() in font_file.stem.lower():
                                return str(font_file)
        
        return None
    
    # Background/text color pairs to simulate diverse real-world conditions
    _color_pairs = [
        ('white', 'black'),           # Standard
        ((245, 245, 245), (30, 30, 30)),  # Light gray bg
        ((240, 235, 220), (50, 40, 30)),  # Beige / parchment
        ((30, 30, 30), (220, 220, 220)),  # Dark bg, light text
        ((0, 0, 0), (255, 255, 255)),     # Pure black bg, white text
        ((235, 240, 250), (20, 40, 80)),  # Light blue bg, dark blue text
        ((255, 250, 240), (60, 60, 60)),  # Warm white bg
        ((50, 50, 70), (200, 200, 220)),  # Dark slate bg
    ]

    def generate_font_sample(self, font_name, text, size=48, output_path=None):
        """
        Generate a sample image for a font.

        Randomly selects a background/text color pair so the trained model
        learns to recognise fonts on varied backgrounds, not just white-on-black.

        Args:
            font_name: Name of the font
            text: Text to render
            size: Font size in pixels
            output_path: Where to save the image
        """
        img_width = 800
        img_height = 200

        # Pick a random background/text colour pair
        bg_color, text_color = random.choice(self._color_pairs)
        img = Image.new('RGB', (img_width, img_height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Try to find and load the font
        font_file = self.find_font_file(font_name)

        if font_file:
            try:
                font = ImageFont.truetype(font_file, size)
            except Exception as e:
                print(f"Warning: Could not load {font_file}, using default font")
                font = ImageFont.load_default()
        else:
            # Try using font name directly (might work for some systems)
            try:
                font = ImageFont.truetype(font_name, size)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
                # Only print warning once per font, not for every sample
                if not hasattr(self, '_font_warnings'):
                    self._font_warnings = set()
                if font_name not in self._font_warnings:
                    print(f"Warning: Font '{font_name}' not found, using default font")
                    print(f"  (Make sure you've run 'python download_google_fonts.py' first)")
                    self._font_warnings.add(font_name)

        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2

        draw.text((x, y), text, fill=text_color, font=font)

        if output_path:
            img.save(output_path)

        return img
    
    def create_font_samples(self, font_name, num_samples=None):
        """
        Create multiple samples for a font with different text and sizes
        
        Args:
            font_name: Name of the font
            num_samples: Number of samples to generate per font (default: 20)
        """
        if num_samples is None:
            num_samples = 20
        font_dir = self.samples_dir / font_name.replace(' ', '_')
        font_dir.mkdir(exist_ok=True)
        
        # Sample texts (common phrases that showcase fonts well)
        sample_texts = [
            "The quick brown fox jumps over the lazy dog",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "abcdefghijklmnopqrstuvwxyz",
            "0123456789",
            "Hello World",
            "Font Detection",
            "Typography Sample",
            "Lorem ipsum dolor sit amet",
            "Sample Text",
            "AaBbCcDdEeFfGg"
        ]
        
        # Font sizes to test
        sizes = [32, 40, 48, 56, 64]
        
        samples_created = []
        
        for i in range(num_samples):
            # Randomly select text and size
            text = random.choice(sample_texts)
            size = random.choice(sizes)
            
            # Generate sample
            output_path = font_dir / f"sample_{i:03d}.png"
            self.generate_font_sample(font_name, text, size, str(output_path))
            samples_created.append({
                "path": str(output_path.relative_to(self.output_dir)),
                "text": text,
                "size": size
            })
        
        return samples_created
    
    def collect_dataset(self):
        """Main method to collect the entire dataset"""
        print(f"Starting font dataset collection...")
        print(f"Target: {self.num_fonts} fonts")
        print(f"Output directory: {self.output_dir}")
        
        metadata = {
            "num_fonts": 0,
            "fonts": []
        }
        
        # Process each font
        for font_name in tqdm(self.target_fonts, desc="Processing fonts"):
            try:
                print(f"\nProcessing: {font_name}")
                
                # Create samples for this font
                samples = self.create_font_samples(font_name, num_samples=getattr(self, '_num_samples', 20))
                
                # Add to metadata
                font_data = {
                    "name": font_name,
                    "samples": samples,
                    "num_samples": len(samples)
                }
                metadata["fonts"].append(font_data)
                metadata["num_fonts"] += 1
                
            except Exception as e:
                print(f"Error processing {font_name}: {e}")
                continue
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset collection complete!")
        print(f"   Fonts collected: {metadata['num_fonts']}")
        print(f"   Total samples: {sum(f['num_samples'] for f in metadata['fonts'])}")
        print(f"   Metadata saved to: {self.metadata_file}")
        
        return metadata

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate font dataset')
    parser.add_argument('--num_fonts', type=int, default=None,
                       help='Number of fonts to use (None = use all available)')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples per font')
    
    args = parser.parse_args()
    
    # Check if downloaded_fonts_list.json exists to determine default
    downloaded_fonts_file = Path("downloaded_fonts_list.json")
    default_num_fonts = 75
    if downloaded_fonts_file.exists():
        try:
            with open(downloaded_fonts_file, 'r') as f:
                downloaded_data = json.load(f)
                available_fonts = downloaded_data.get('available', [])
                if available_fonts:
                    # If we have downloaded fonts, use all of them by default
                    default_num_fonts = len(available_fonts)
                    print(f"Found {len(available_fonts)} downloaded fonts - will use all by default")
        except:
            pass
    
    collector = FontDatasetCollector(
        output_dir="font_dataset",
        num_fonts=args.num_fonts if args.num_fonts else default_num_fonts,
        num_samples=args.num_samples
    )
    
    metadata = collector.collect_dataset()
    
    print("\n" + "="*50)
    print("Phase 1 Complete: Dataset Ready")
    print("="*50)
    print("\nNext steps:")
    print("1. Review generated samples in font_dataset/samples/")
    print("2. Verify font quality and diversity")
    print("3. Proceed to Phase 2: Model Training")

if __name__ == "__main__":
    main()

