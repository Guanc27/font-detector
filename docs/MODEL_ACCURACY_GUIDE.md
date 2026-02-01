# Model Accuracy Guide

## Expected Accuracy

### Baseline Expectations (MVP)

With your current setup (75 fonts, ~750 samples, 10 epochs):

| Metric | Expected Range | Good Performance |
|--------|---------------|------------------|
| **Top-1 Accuracy** | 40-60% | 60%+ |
| **Top-3 Accuracy** | 60-80% | 80%+ |
| **Top-5 Accuracy** | 70-90% | 90%+ |

**Why Top-K matters**: For font detection, getting the correct font in the top 3-5 candidates is often acceptable, especially since fonts can look similar.

### Realistic Expectations

- **Clean images** (like your training data): 60-80% top-1 accuracy
- **Real-world images** (with noise, blur, etc.): 40-60% top-1 accuracy
- **After data augmentation**: 50-70% top-1 on real-world images

## Factors Affecting Accuracy

### 1. Dataset Size
- **Current**: ~750 samples (10 per font)
- **More samples** = Better accuracy
- **Recommendation**: 20-30 samples per font for better results

### 2. Model Architecture
- **ViT-B-32** (current): Good balance
- **ViT-B-16**: Faster, slightly less accurate
- **ViT-L-14**: More accurate, slower, needs more memory

### 3. Training Duration
- **5 epochs**: Quick test, ~40-50% accuracy
- **10 epochs**: Good for MVP, ~50-60% accuracy
- **20-30 epochs**: Better accuracy, ~60-70% accuracy

### 4. Data Quality
- **Clean samples**: Higher accuracy
- **Diverse samples**: Better generalization
- **Augmented data**: More robust to real-world conditions

## How to Increase Accuracy

### Method 1: More Training Data ✅ (Easiest)

**Increase samples per font:**

```python
# In data_collection.py, change:
num_samples=20  # Instead of 10
```

**Benefits:**
- More diverse examples
- Better generalization
- +10-15% accuracy improvement

### Method 2: Data Augmentation ✅ (Best for Robustness)

Add transformations to simulate real-world conditions:

- **Blur**: Simulates out-of-focus images
- **Noise**: Simulates compression artifacts
- **Rotation**: Handles tilted text
- **Brightness/Contrast**: Handles lighting variations
- **Distortion**: Simulates perspective/warping

**Expected improvement**: +15-25% accuracy on real-world images

### Method 3: Longer Training

```powershell
python train_embedding_model.py --epochs 20 --lr 5e-5
```

**Benefits:**
- Model learns better features
- +5-10% accuracy improvement
- Diminishing returns after 20-30 epochs

### Method 4: Better Model Architecture

```powershell
python train_embedding_model.py --model ViT-L-14
```

**Benefits:**
- More capacity for complex patterns
- +10-15% accuracy improvement
- Requires more memory/time

### Method 5: Learning Rate Tuning

```powershell
# Lower learning rate (more careful updates)
python train_embedding_model.py --lr 5e-5

# Higher learning rate (faster learning, riskier)
python train_embedding_model.py --lr 5e-4
```

### Method 6: Ensemble Methods

Train multiple models and combine predictions:
- Train 3-5 models with different seeds
- Average their predictions
- +5-10% accuracy improvement

## Data Augmentation Strategy

### Why Augmentation Matters

Real-world font images have:
- ❌ Blur (camera focus issues)
- ❌ Noise (compression artifacts)
- ❌ Rotation (tilted documents)
- ❌ Lighting variations
- ❌ Perspective distortion

### Augmentation Types

1. **Geometric Transformations**
   - Rotation (±5-10 degrees)
   - Translation (slight shifts)
   - Scaling (95-105%)

2. **Photometric Transformations**
   - Brightness (±20%)
   - Contrast (±20%)
   - Saturation adjustments

3. **Noise & Blur**
   - Gaussian blur (simulates focus issues)
   - Gaussian noise (compression artifacts)
   - Salt & pepper noise

4. **Distortion**
   - Elastic deformation (subtle warping)
   - Perspective transform (document scanning)

### Recommended Augmentation Pipeline

```python
transforms.Compose([
    # Geometric
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    
    # Photometric
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    
    # Blur (applied randomly)
    transforms.RandomApply([GaussianBlur()], p=0.3),
    
    # Noise (applied randomly)
    transforms.RandomApply([AddGaussianNoise()], p=0.3),
    
    # Standard preprocessing
    preprocess
])
```

## Accuracy Improvement Roadmap

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Add data augmentation → +15-20% accuracy
2. ✅ Train for 15 epochs → +5-10% accuracy
3. **Result**: ~60-70% top-1 accuracy

### Phase 2: Better Data (2-4 hours)
1. ✅ Generate 20 samples per font → +10-15% accuracy
2. ✅ Add more diverse text samples → +5% accuracy
3. **Result**: ~70-80% top-1 accuracy

### Phase 3: Advanced (4-8 hours)
1. ✅ Use larger model (ViT-L-14) → +10-15% accuracy
2. ✅ Train ensemble → +5-10% accuracy
3. ✅ Fine-tune hyperparameters → +5% accuracy
4. **Result**: ~80-90% top-1 accuracy

## Measuring Accuracy

The training script reports:
- **Top-1 Accuracy**: Correct font is #1 prediction
- **Top-3 Accuracy**: Correct font in top 3 predictions
- **Top-5 Accuracy**: Correct font in top 5 predictions

For font detection, **Top-3 accuracy** is often more meaningful than Top-1.

## Real-World Performance

### Clean Images (like training data)
- Expected: 60-80% top-1
- With augmentation: 70-85% top-1

### Real-World Images (with issues)
- Expected: 40-60% top-1
- With augmentation: 55-75% top-1

### Why the Gap?
- Real images have unpredictable conditions
- Training data is clean and controlled
- Augmentation helps bridge this gap

## Recommendations for MVP

**Minimum viable setup:**
- ✅ Current dataset (750 samples)
- ✅ Add data augmentation
- ✅ Train 15 epochs
- **Expected**: 55-65% top-1, 75-85% top-3

**Better setup:**
- ✅ 20 samples per font (1500 samples)
- ✅ Data augmentation
- ✅ Train 20 epochs
- **Expected**: 65-75% top-1, 85-90% top-3

**Best setup (post-MVP):**
- ✅ 30+ samples per font
- ✅ Advanced augmentation
- ✅ Larger model
- ✅ Ensemble
- **Expected**: 75-85% top-1, 90-95% top-3

