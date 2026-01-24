# Parameters Guide

This guide explains all the parameters you can use in the font detection project scripts.

## Data Collection Parameters (`data_collection.py`)

### `output_dir` (default: `"font_dataset"`)
**What it does**: Where to save the generated font samples

**Example**:
```python
collector = FontDatasetCollector(output_dir="my_font_dataset")
```

**When to change**: If you want to keep multiple datasets or organize differently

---

### `num_fonts` (default: `75`)
**What it does**: How many fonts to process from the target list

**Example**:
```python
collector = FontDatasetCollector(num_fonts=50)  # Use only first 50 fonts
```

**When to change**: 
- Testing: Use fewer fonts (10-20) for quick tests
- Production: Use all 75 or add more fonts

---

### `num_samples` (default: `20` in code, `10` was original)
**What it does**: Number of sample images to generate per font

**Example**:
```python
samples = self.create_font_samples(font_name, num_samples=30)
```

**Impact**:
- **10 samples**: Fast, ~750 total samples, basic accuracy
- **20 samples**: Balanced, ~1,500 total samples, good accuracy
- **30 samples**: Best, ~2,250 total samples, highest accuracy

**When to change**: 
- More samples = Better accuracy but longer generation time
- Recommended: 20 for MVP, 30+ for production

---

## Training Parameters (`train_embedding_model.py`)

### `--dataset_dir` (default: `"font_dataset"`)
**What it does**: Path to your font dataset directory

**Example**:
```powershell
python train_embedding_model.py --dataset_dir my_font_dataset
```

**When to change**: If you saved your dataset in a different location

---

### `--metadata` (default: `"font_dataset/metadata.json"`)
**What it does**: Path to the metadata JSON file

**Example**:
```powershell
python train_embedding_model.py --metadata my_font_dataset/metadata.json
```

**When to change**: If metadata file is in a different location

---

### `--model` (default: `"ViT-B-32"`)
**What it does**: Which OpenCLIP model architecture to use

**Available options**:
- `ViT-B-16`: Smaller, faster, less accurate (~512MB)
- `ViT-B-32`: **Recommended** - Balanced (~600MB)
- `ViT-L-14`: Larger, slower, more accurate (~1.2GB)
- `ViT-H-14`: Huge, very slow, best accuracy (~2GB)

**Example**:
```powershell
python train_embedding_model.py --model ViT-L-14
```

**When to change**:
- **Smaller model** (`ViT-B-16`): If you have limited GPU memory or want faster training
- **Larger model** (`ViT-L-14`): If you have GPU and want better accuracy

**Trade-offs**:
| Model | Size | Speed | Accuracy | Memory |
|-------|------|-------|----------|--------|
| ViT-B-16 | Small | Fast | Good | Low |
| ViT-B-32 | Medium | Medium | Better | Medium |
| ViT-L-14 | Large | Slow | Best | High |

---

### `--pretrained` (default: `"openai"`)
**What it does**: Which pretrained weights to use

**Available options**:
- `"openai"`: OpenAI's pretrained CLIP weights (recommended)
- `"laion400m"`: Trained on LAION-400M dataset
- `"laion2b"`: Trained on LAION-2B dataset (larger)

**Example**:
```powershell
python train_embedding_model.py --pretrained laion400m
```

**When to change**: Usually keep `"openai"` - it's the most reliable

---

### `--epochs` (default: `10`)
**What it does**: How many times to train on the entire dataset

**Example**:
```powershell
python train_embedding_model.py --epochs 20
```

**Impact**:
- **5 epochs**: Quick test, ~40-50% accuracy
- **10 epochs**: Good for MVP, ~50-60% accuracy
- **20 epochs**: Better accuracy, ~60-70% accuracy
- **30+ epochs**: Diminishing returns, risk of overfitting

**When to change**:
- Start with 10 for testing
- Increase to 20-30 for better results
- Watch validation accuracy - if it stops improving, stop training

**Rule of thumb**: More epochs = better accuracy, but takes longer

---

### `--batch_size` (default: `32`)
**What it does**: How many images to process at once

**Example**:
```powershell
python train_embedding_model.py --batch_size 64
```

**Impact**:
- **Smaller batch** (8-16): Uses less memory, slower training
- **Larger batch** (32-64): Uses more memory, faster training

**When to change**:
- **Reduce** if you get "Out of Memory" errors:
  ```powershell
  python train_embedding_model.py --batch_size 16
  ```
- **Increase** if you have GPU with lots of memory:
  ```powershell
  python train_embedding_model.py --batch_size 64
  ```

**Memory guide**:
- CPU: Use 8-16
- GPU (4GB): Use 16-32
- GPU (8GB+): Use 32-64

---

### `--lr` (default: `1e-4` = `0.0001`)
**What it does**: Learning rate - how big steps the model takes when learning

**Example**:
```powershell
python train_embedding_model.py --lr 5e-5  # Smaller steps, more careful
python train_embedding_model.py --lr 5e-4  # Larger steps, faster learning
```

**Impact**:
- **Too high** (`1e-3`): Model might not converge, unstable training
- **Too low** (`1e-5`): Very slow learning, might not reach good accuracy
- **Just right** (`1e-4` to `5e-4`): Good balance

**When to change**:
- **Lower** (`5e-5`): If training is unstable or accuracy plateaus early
- **Higher** (`5e-4`): If training is very slow and accuracy improves slowly

**Rule of thumb**: Start with default, adjust if training doesn't improve

---

### `--save_dir` (default: `"models"`)
**What it does**: Where to save trained model checkpoints

**Example**:
```powershell
python train_embedding_model.py --save_dir my_models
```

**When to change**: If you want to organize models differently or keep multiple versions

---

### `--no_augment` (flag, default: `False`)
**What it does**: Disable data augmentation (blur, noise, rotation, etc.)

**Example**:
```powershell
python train_embedding_model.py --no_augment
```

**When to use**: 
- For comparison/testing
- If you want clean training data only
- **Not recommended** for production - augmentation improves robustness

---

## Common Parameter Combinations

### Quick Test (Fast, Lower Accuracy)
```powershell
python train_embedding_model.py --epochs 5 --batch_size 16 --model ViT-B-16
```
**Use for**: Testing if everything works

---

### MVP Setup (Balanced)
```powershell
python train_embedding_model.py --epochs 10 --batch_size 32 --model ViT-B-32
```
**Use for**: Your MVP - good balance of speed and accuracy

---

### Best Accuracy (Slower, Better Results)
```powershell
python train_embedding_model.py --epochs 20 --batch_size 32 --model ViT-L-14 --lr 5e-5
```
**Use for**: Production - maximum accuracy

---

### Limited Memory (CPU or Small GPU)
```powershell
python train_embedding_model.py --epochs 10 --batch_size 8 --model ViT-B-16
```
**Use for**: If you get out-of-memory errors

---

## Understanding Parameter Interactions

### Model Size ↔ Batch Size
- **Larger model** = Need smaller batch size (less memory available)
- **Smaller model** = Can use larger batch size (more memory available)

### Epochs ↔ Learning Rate
- **More epochs** = Can use lower learning rate (more careful learning)
- **Fewer epochs** = Might need higher learning rate (learn faster)

### Batch Size ↔ Training Speed
- **Larger batch** = Faster training (more parallel processing)
- **Smaller batch** = Slower training (less parallel processing)

## Parameter Tuning Tips

1. **Start with defaults**: Most parameters work well out of the box
2. **Change one at a time**: So you know what affects what
3. **Monitor validation accuracy**: Stop if it stops improving
4. **Watch for overfitting**: If train accuracy >> val accuracy, reduce epochs or add regularization

## Quick Reference Table

| Parameter | Default | What It Controls | Increase When | Decrease When |
|-----------|---------|------------------|---------------|---------------|
| `epochs` | 10 | Training duration | Want better accuracy | Quick testing |
| `batch_size` | 32 | Memory usage | Have GPU memory | Out of memory |
| `lr` | 1e-4 | Learning speed | Training too slow | Unstable training |
| `model` | ViT-B-32 | Model capacity | Want accuracy | Limited memory |
| `num_samples` | 20 | Dataset size | Want accuracy | Quick testing |

## Need Help?

- **Out of memory?** → Reduce `batch_size` or use smaller `model`
- **Training too slow?** → Increase `batch_size` or reduce `epochs`
- **Low accuracy?** → Increase `epochs`, `num_samples`, or use larger `model`
- **Unstable training?** → Reduce `lr` or `batch_size`

