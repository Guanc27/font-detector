# Phase 2: Embedding Model Training Guide

## Overview

Phase 2 fine-tunes OpenCLIP (or CLIP) on your font dataset to create font-specific embeddings. This allows the model to understand font characteristics better than a generic vision model.

## Prerequisites

✅ Phase 1 Complete:
- Font dataset created (`font_dataset/` directory)
- `metadata.json` exists
- Font samples generated

## Step 1: Install Dependencies

Install the new ML dependencies:

```powershell
pip install torch torchvision open-clip-torch
```

**Note**: PyTorch installation can be large (~2GB). If you have a GPU, install CUDA version:
```powershell
# For CUDA (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only (slower but works)
pip install torch torchvision
```

Then install OpenCLIP:
```powershell
pip install open-clip-torch
```

## Step 2: Verify Dataset

Make sure your dataset is ready:

```powershell
python -c "import json; data=json.load(open('font_dataset/metadata.json')); print(f'Fonts: {data[\"num_fonts\"]}, Samples: {sum(f[\"num_samples\"] for f in data[\"fonts\"])}')"
```

## Step 3: Run Training

### Basic Training (Recommended for MVP):

```powershell
python train_embedding_model.py --epochs 5 --batch_size 16
```

### Full Training (Better Results):

```powershell
python train_embedding_model.py --epochs 10 --batch_size 32 --lr 1e-4
```

### With GPU (if available):

```powershell
python train_embedding_model.py --epochs 10 --batch_size 64
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 32 | Batch size (reduce if out of memory) |
| `--lr` | 1e-4 | Learning rate |
| `--model` | ViT-B-32 | Model architecture |
| `--pretrained` | openai | Pretrained weights |

## What Happens During Training

1. **Model Loading**: Loads OpenCLIP (or falls back to CLIP)
2. **Freezing**: Freezes most of the pretrained model
3. **Fine-tuning**: Unfreezes last layers for adaptation
4. **Training**: Trains classifier head on font dataset
5. **Validation**: Evaluates on validation set each epoch
6. **Saving**: Saves best model checkpoint

## Expected Output

```
Phase 2: Font Embedding Model Training
============================================================

1. Loading dataset...
Loaded 750 samples from 75 fonts

2. Splitting dataset...
Dataset splits:
  Train: 525 samples
  Val: 112 samples
  Test: 113 samples

3. Initializing model...
Using device: cuda (or cpu)
Loading OpenCLIP model: ViT-B-32...
Model setup complete:
  - Vision encoder: Mostly frozen (last layers trainable)
  - Classifier head: Trainable

4. Training model...
Epoch 1/10 [Train]: 100%|████| 17/17 [00:45<00:00, loss=2.1234, acc=45.23%]
Validating: 100%|████| 4/4 [00:08<00:00]

Epoch 1/10:
  Train Loss: 2.1234, Train Acc: 45.23%
  Val Loss: 1.9876, Val Acc: 52.34%
  ✅ Saved best model (Val Acc: 52.34%)

...
```

## Training Time Estimates

| Hardware | Batch Size | Epochs | Estimated Time |
|----------|------------|--------|----------------|
| CPU | 16 | 10 | 2-4 hours |
| CPU | 32 | 10 | 1-2 hours |
| GPU (GTX 1060) | 32 | 10 | 30-60 min |
| GPU (RTX 3080) | 64 | 10 | 15-30 min |

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```powershell
python train_embedding_model.py --batch_size 8
```

### OpenCLIP Not Found

Install it:
```powershell
pip install open-clip-torch
```

The script will automatically fall back to OpenAI CLIP if OpenCLIP isn't available.

### Slow Training

- Use smaller model: `--model ViT-B-16`
- Reduce epochs: `--epochs 5`
- Use GPU if available

### Low Accuracy

- Train for more epochs: `--epochs 20`
- Increase learning rate: `--lr 5e-4`
- Check dataset quality (run Phase 1 again)

## Output Files

After training, you'll have:

```
models/
└── best_model.pt  # Trained model checkpoint
```

This checkpoint contains:
- Model weights
- Classifier head
- Optimizer state
- Validation accuracy

## Next Steps

After training completes:

1. ✅ Model saved to `models/best_model.pt`
2. ✅ Ready for Phase 3: Vector Database Setup
3. ✅ Use model to generate embeddings for all fonts

## Quick Test

Test that training works (small test run):

```powershell
python train_embedding_model.py --epochs 1 --batch_size 8
```

This runs 1 epoch to verify everything works before full training.

## Tips

- **Start small**: Test with 1-2 epochs first
- **Monitor GPU**: Use `nvidia-smi` to check GPU usage
- **Save checkpoints**: Model saves automatically after each epoch
- **Be patient**: Training takes time, especially on CPU

Good luck! 🚀

