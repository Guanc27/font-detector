"""
Phase 2: Font Embedding Model Training
Fine-tunes OpenCLIP on font dataset to create font embeddings
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import random

# Try to import OpenCLIP, fallback to CLIP
try:
    import open_clip
    HAS_OPENCLIP = True
except ImportError:
    HAS_OPENCLIP = False
    try:
        import clip
        HAS_CLIP = True
    except ImportError:
        HAS_CLIP = False
        raise ImportError("Please install either 'open-clip-torch' or 'clip' package")

class GaussianBlur:
    """Apply Gaussian blur to image"""
    def __init__(self, radius_range=(0.5, 2.0)):
        self.radius_range = radius_range
    
    def __call__(self, img):
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class AddGaussianNoise:
    """Add Gaussian noise to image"""
    def __init__(self, mean=0, std_range=(5, 25)):
        self.mean = mean
        self.std_range = std_range
    
    def __call__(self, img):
        std = random.uniform(*self.std_range)
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(self.mean, std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

class RandomPerspective:
    """Apply random perspective transformation"""
    def __init__(self, distortion_scale=0.1, p=0.3):
        self.distortion_scale = distortion_scale
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            w, h = img.size
            # Create perspective transform
            startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
            endpoints = []
            for x, y in startpoints:
                dx = random.uniform(-w * self.distortion_scale, w * self.distortion_scale)
                dy = random.uniform(-h * self.distortion_scale, h * self.distortion_scale)
                endpoints.append([x + dx, y + dy])
            
            return img.transform(
                img.size,
                Image.Transform.PERSPECTIVE,
                self._get_perspective_coeffs(startpoints, endpoints),
                Image.Resampling.BILINEAR
            )
        return img
    
    def _get_perspective_coeffs(self, startpoints, endpoints):
        """Calculate perspective transform coefficients"""
        matrix = []
        for p1, p2 in zip(startpoints, endpoints):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.array(matrix, dtype=np.float32)
        B = np.array(endpoints, dtype=np.float32).reshape(8)
        res = np.linalg.lstsq(A, B, rcond=None)[0]
        return np.concatenate([res, [1.0]]).reshape(9)

class FontDataset(Dataset):
    """
    Dataset for loading font samples with labels
    """
    def __init__(self, dataset_dir, metadata_file, transform=None, augment=False):
        """
        Args:
            dataset_dir: Path to font_dataset directory
            metadata_file: Path to metadata.json
            transform: Optional image transforms
            augment: Whether to apply data augmentation
        """
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.augment = augment
        
        # Load metadata
        print(f"Loading metadata from: {metadata_file}")
        print(f"Dataset directory: {dataset_dir}")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"Metadata loaded: {len(metadata.get('fonts', []))} fonts")
        
        # Create samples list: (image_path, font_label)
        self.samples = []
        self.font_to_idx = {}
        self.idx_to_font = {}
        
        # Debug: Check dataset directory
        dataset_path = Path(dataset_dir)
        print(f"Dataset path exists: {dataset_path.exists()}")
        print(f"Dataset path absolute: {dataset_path.absolute()}")
        if dataset_path.exists():
            print(f"Contents: {list(dataset_path.iterdir())[:10]}")
        
        font_idx = 0
        missing_count = 0
        found_count = 0
        
        for font_data in metadata['fonts']:
            font_name = font_data['name']
            
            # Create mapping
            if font_name not in self.font_to_idx:
                self.font_to_idx[font_name] = font_idx
                self.idx_to_font[font_idx] = font_name
                font_idx += 1
            
            label = self.font_to_idx[font_name]
            
            # Add all samples for this font
            for sample in font_data['samples']:
                # Fix Windows backslashes to forward slashes
                sample_path = sample['path'].replace('\\', '/')
                image_path = self.dataset_dir / sample_path
                
                if image_path.exists():
                    self.samples.append((str(image_path), label))
                    found_count += 1
                else:
                    missing_count += 1
                    # Debug: print first few missing files
                    if missing_count <= 5:
                        print(f"\n⚠️  Image not found:")
                        print(f"  Expected: {image_path.absolute()}")
                        print(f"  Dataset dir: {self.dataset_dir.absolute()}")
                        print(f"  Path from metadata (original): {sample['path']}")
                        print(f"  Path from metadata (fixed): {sample_path}")
                        # Try to find where it actually is
                        if (self.dataset_dir / 'samples').exists():
                            font_folders = list((self.dataset_dir / 'samples').iterdir())
                            print(f"  Found {len(font_folders)} font folders in samples/")
                            if font_folders:
                                first_font_samples = list(font_folders[0].iterdir())
                                print(f"  First font has {len(first_font_samples)} files")
                                if first_font_samples:
                                    print(f"  Example file: {first_font_samples[0]}")
        
        print(f"\nFile check summary:")
        print(f"  Found: {found_count} files")
        print(f"  Missing: {missing_count} files")
        
        self.num_fonts = len(self.font_to_idx)
        print(f"Loaded {len(self.samples)} samples from {self.num_fonts} fonts")
        
        if len(self.samples) == 0:
            print("\n⚠️  ERROR: No samples found!")
            print(f"Dataset directory: {self.dataset_dir.absolute()}")
            print(f"Metadata file: {metadata_file}")
            print(f"Checking if dataset directory exists: {self.dataset_dir.exists()}")
            if self.dataset_dir.exists():
                print(f"Contents of dataset dir: {list(self.dataset_dir.iterdir())[:10]}")
            print("\nTroubleshooting:")
            print("1. Check that --dataset_dir points to the folder containing 'samples' and 'metadata.json'")
            print("2. Check that metadata.json paths match actual file locations")
            print("3. Try using absolute paths")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply augmentation if enabled (before standard transform)
        if self.augment:
            image = self._apply_augmentation(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _apply_augmentation(self, image):
        """Apply random augmentations to simulate real-world conditions"""
        # Random rotation (±5 degrees)
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, fillcolor='white', resample=Image.BILINEAR)
        
        # Random brightness/contrast
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Random blur (simulates focus issues)
        if random.random() < 0.3:
            blur = GaussianBlur(radius_range=(0.5, 1.5))
            image = blur(image)
        
        # Random noise (simulates compression artifacts)
        if random.random() < 0.3:
            noise = AddGaussianNoise(std_range=(5, 15))
            image = noise(image)
        
        # Random perspective (subtle, simulates scanning)
        if random.random() < 0.2:
            perspective = RandomPerspective(distortion_scale=0.05, p=1.0)
            image = perspective(image)
        
        return image

class FontEmbeddingTrainer:
    """
    Trainer for fine-tuning OpenCLIP on font dataset
    """
    def __init__(self, model_name='ViT-B-32', pretrained='openai', device=None):
        """
        Initialize the trainer
        
        Args:
            model_name: OpenCLIP model architecture
            pretrained: Pretrained weights
            device: torch device (auto-detected if None)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load OpenCLIP model
        print(f"Loading OpenCLIP model: {model_name}...")
        self.model, self.preprocess, self.tokenizer, self.use_openclip = self._load_model(model_name, pretrained)
        self.model = self.model.to(self.device)
        
        # Freeze base model, add trainable projection head
        self._setup_model_for_finetuning()
        
    def _load_model(self, model_name, pretrained):
        """Load OpenCLIP or CLIP model"""
        if HAS_OPENCLIP:
            try:
                # Try OpenCLIP first (more features)
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained=pretrained,
                    device=self.device
                )
                tokenizer = open_clip.get_tokenizer(model_name)
                return model, preprocess, tokenizer, True
            except Exception as e:
                print(f"Error loading OpenCLIP: {e}")
                if HAS_CLIP:
                    print("Falling back to OpenAI CLIP...")
                else:
                    raise
        
        if HAS_CLIP:
            try:
                # Fallback to OpenAI CLIP
                model, preprocess = clip.load("ViT-B/32", device=self.device)
                tokenizer = None
                return model, preprocess, tokenizer, False
            except Exception as e2:
                print(f"Error loading CLIP: {e2}")
                raise RuntimeError("Could not load any CLIP model. Please install open-clip-torch or clip")
        
        raise RuntimeError("No CLIP library available. Please install open-clip-torch or clip")
    
    def _setup_model_for_finetuning(self):
        """Setup model for fine-tuning"""
        # Freeze vision encoder (keep pretrained features)
        for param in self.model.visual.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers for fine-tuning
        # This allows adaptation while keeping pretrained features
        if hasattr(self.model.visual, 'transformer'):
            # Unfreeze last transformer block
            for param in self.model.visual.transformer.resblocks[-2:].parameters():
                param.requires_grad = True
        
        # Add projection head for font classification
        embedding_dim = self.model.visual.output_dim if hasattr(self.model.visual, 'output_dim') else 512
        
        # We'll use the model's existing projection, but add a classifier head
        self.classifier = nn.Linear(embedding_dim, 1000).to(self.device)  # Will resize based on num_fonts
        self.classifier.requires_grad = True
        
        print("Model setup complete:")
        print(f"  - Vision encoder: Mostly frozen (last layers trainable)")
        print(f"  - Classifier head: Trainable")
    
    def train(self, train_loader, val_loader, num_fonts, epochs=10, lr=1e-4, save_dir='models'):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_fonts: Number of font classes
            epochs: Number of training epochs
            lr: Learning rate
            save_dir: Directory to save checkpoints
        """
        # Resize classifier to match number of fonts
        embedding_dim = self.classifier.in_features
        self.classifier = nn.Linear(embedding_dim, num_fonts).to(self.device)
        
        # Setup optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_params.extend(self.classifier.parameters())
        
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        best_val_acc = 0.0
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Learning rate: {lr}")
        print(f"Number of fonts: {num_fonts}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                # Get image embeddings
                if self.use_openclip:
                    image_features = self.model.encode_image(images)
                else:
                    # OpenAI CLIP
                    image_features = self.model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Classify
                logits = self.classifier(image_features)
                loss = criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*train_correct/train_total:.2f}%'
                })
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_acc, val_loss = self.validate(val_loader, criterion)
            
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'classifier_state_dict': self.classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'num_fonts': num_fonts,
                }
                torch.save(checkpoint, save_path / 'best_model.pt')
                print(f"  ✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get embeddings and classify
                if self.use_openclip:
                    image_features = self.model.encode_image(images)
                else:
                    # OpenAI CLIP
                    image_features = self.model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                logits = self.classifier(image_features)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = val_loss / len(val_loader)
        return accuracy, avg_loss
    
    def get_embeddings(self, images):
        """
        Get embeddings for images (for inference)
        
        Args:
            images: Batch of images
            
        Returns:
            Embeddings tensor
        """
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            if self.use_openclip:
                embeddings = self.model.encode_image(images)
            else:
                embeddings = self.model.encode_image(images)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset into train/val/test
    
    Args:
        dataset: FontDataset instance
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from torch.utils.data import Subset
    import random
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Shuffle indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Calculate splits
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset

def main():
    """Main training execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train font embedding model')
    parser.add_argument('--dataset_dir', type=str, default='font_dataset',
                       help='Path to font dataset directory')
    parser.add_argument('--metadata', type=str, default='font_dataset/metadata.json',
                       help='Path to metadata.json')
    parser.add_argument('--model', type=str, default='ViT-B-32',
                       help='OpenCLIP model name')
    parser.add_argument('--pretrained', type=str, default='openai',
                       help='Pretrained weights')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--no_augment', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 2: Font Embedding Model Training")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = FontDataset(args.dataset_dir, args.metadata, transform=None)
    
    # Split dataset
    print("\n2. Splitting dataset...")
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    
    # Create trainer first to get preprocessing
    print("\n3. Initializing model...")
    trainer = FontEmbeddingTrainer(model_name=args.model, pretrained=args.pretrained)
    
    # Apply preprocessing to datasets
    transform = trainer.preprocess
    
    # Update datasets with transforms
    # Use augmentation for training, not for validation
    train_dataset.dataset.transform = transform
    train_dataset.dataset.augment = not args.no_augment  # Enable augmentation unless disabled
    
    val_dataset.dataset.transform = transform
    val_dataset.dataset.augment = False  # No augmentation for validation
    
    if not args.no_augment:
        print("\n✅ Data augmentation enabled (blur, noise, rotation, etc.)")
    else:
        print("\n⚠️  Data augmentation disabled")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Train model
    print("\n4. Training model...")
    best_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_fonts=dataset.num_fonts,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir
    )
    
    print("\n" + "="*60)
    print("Phase 2 Complete!")
    print("="*60)
    print(f"\nModel saved to: {args.save_dir}/best_model.pt")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print("\nNext steps:")
    print("1. Proceed to Phase 3: Vector Database Setup")
    print("2. Generate embeddings for all font samples")

if __name__ == "__main__":
    main()

