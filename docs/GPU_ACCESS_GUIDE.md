# GPU Access Guide - Training Your Model Without a GPU

## Overview

You don't need to own a GPU to train your model! There are many ways to access GPU computing power, from free options to paid cloud services.

## Free GPU Options (Best for MVP)

### 1. Google Colab ⭐ **RECOMMENDED**

**What it is**: Free Jupyter notebook environment with free GPU access

**GPU Access**:
- ✅ **Free tier**: T4 GPU (16GB) for ~12 hours/day
- ✅ **Colab Pro**: $10/month - Better GPUs, longer sessions
- ✅ **Colab Pro+**: $20/month - Priority access, best GPUs

**How to use**:
1. Go to: https://colab.research.google.com/
2. Create new notebook
3. Runtime → Change runtime type → GPU (T4)
4. Upload your code and data
5. Run training!

**Pros**:
- ✅ Completely free
- ✅ Easy to use (browser-based)
- ✅ Pre-installed ML libraries
- ✅ Can upload files directly

**Cons**:
- ⚠️ Sessions timeout after ~12 hours
- ⚠️ Files deleted when session ends (save to Google Drive!)
- ⚠️ Free tier has usage limits

**Setup Steps**:
```python
# In Colab notebook:
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies
!pip install torch torchvision open-clip-torch

# 3. Upload your code
# (or clone from GitHub)

# 4. Run training
!python train_embedding_model.py --epochs 10 --batch_size 32
```

**Best for**: MVP, testing, learning

---

### 2. Kaggle Notebooks

**What it is**: Free GPU access for data science competitions

**GPU Access**:
- ✅ **Free**: 30 hours/week of GPU time
- ✅ P100 GPU (16GB)
- ✅ No credit card required

**How to use**:
1. Go to: https://www.kaggle.com/
2. Create account
3. New Notebook → Settings → GPU enabled
4. Upload dataset and code

**Pros**:
- ✅ 30 hours/week free
- ✅ Good GPU (P100)
- ✅ Pre-installed libraries

**Cons**:
- ⚠️ Weekly time limit
- ⚠️ Must use Kaggle's interface

**Best for**: Regular training sessions

---

### 3. Paperspace Gradient

**What it is**: Free GPU instances for ML

**GPU Access**:
- ✅ **Free tier**: Limited hours/month
- ✅ Paid: $0.51/hour for GPU instances

**How to use**:
1. Sign up at: https://www.paperspace.com/
2. Create Gradient notebook
3. Select GPU instance
4. Run your code

**Pros**:
- ✅ Free tier available
- ✅ Good GPU options
- ✅ Jupyter notebook interface

**Cons**:
- ⚠️ Free tier is limited
- ⚠️ Paid tier charges by hour

**Best for**: Occasional training

---

## Paid Cloud GPU Options

### 1. Google Cloud Platform (GCP)

**Pricing**: ~$0.50-2.00/hour depending on GPU

**GPUs Available**:
- T4: $0.35/hour
- V100: $2.48/hour
- A100: $3.67/hour

**How to use**:
1. Sign up: https://cloud.google.com/
2. Create Compute Engine instance
3. Select GPU
4. SSH in and run training

**Pros**:
- ✅ Pay only for what you use
- ✅ Powerful GPUs available
- ✅ $300 free credit for new users

**Cons**:
- ⚠️ Requires setup
- ⚠️ Can get expensive if left running

---

### 2. AWS EC2

**Pricing**: ~$0.50-4.00/hour

**GPUs Available**:
- g4dn.xlarge: $0.526/hour (T4)
- p3.2xlarge: $3.06/hour (V100)

**How to use**:
1. Sign up: https://aws.amazon.com/
2. Launch EC2 instance
3. Select GPU instance type
4. Connect and run training

**Pros**:
- ✅ Industry standard
- ✅ Many GPU options
- ✅ $300 free credit for new users

**Cons**:
- ⚠️ More complex setup
- ⚠️ Easy to forget to stop instances (costs money!)

---

### 3. Azure

**Pricing**: Similar to AWS/GCP

**GPUs Available**:
- NC6s_v3: ~$0.90/hour
- ND6s: ~$2.00/hour

**Pros**:
- ✅ $200 free credit for new users
- ✅ Good integration with Microsoft tools

**Cons**:
- ⚠️ Similar complexity to AWS/GCP

---

## Comparison Table

| Service | Free Tier | Paid/Hour | GPU Type | Best For |
|---------|-----------|-----------|----------|----------|
| **Google Colab** | ✅ Yes (12h/day) | $10-20/mo | T4 | **MVP, Learning** ⭐ |
| **Kaggle** | ✅ Yes (30h/week) | Free | P100 | Regular training |
| **Paperspace** | ⚠️ Limited | $0.51/hr | Various | Occasional use |
| **GCP** | $300 credit | $0.35-3.67/hr | T4/V100/A100 | Production |
| **AWS** | $300 credit | $0.53-3.06/hr | T4/V100 | Production |
| **Azure** | $200 credit | $0.90-2.00/hr | Various | Enterprise |

## Recommended Approach for Your MVP

### Option 1: Google Colab (Easiest) ⭐

**Steps**:
1. **Prepare your code**:
   - Make sure it works locally first
   - Upload to GitHub or Google Drive

2. **Create Colab notebook**:
   ```python
   # Cell 1: Mount Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Cell 2: Install dependencies
   !pip install torch torchvision open-clip-torch tqdm pillow
   
   # Cell 3: Navigate to your project
   %cd /content/drive/MyDrive/check_fonts
   
   # Cell 4: Run training
   !python train_embedding_model.py --epochs 10 --batch_size 32
   ```

3. **Upload your dataset**:
   - Upload `font_dataset/` folder to Google Drive
   - Or use Colab's file upload feature

4. **Run and monitor**:
   - Training will show progress in notebook
   - Download model when done

**Time estimate**: 30-60 minutes for 10 epochs

---

### Option 2: Kaggle (More Stable)

**Steps**:
1. Create Kaggle account
2. Upload dataset as Kaggle dataset
3. Create notebook with GPU enabled
4. Clone your code or write in notebook
5. Run training

**Advantage**: More stable than Colab, 30 hours/week

---

## Adapting Your Code for Cloud GPUs

### Changes Needed

**1. Path handling**:
```python
# Instead of relative paths:
dataset_dir = "font_dataset"

# Use absolute paths or environment variables:
import os
dataset_dir = os.getenv("DATASET_DIR", "/content/drive/MyDrive/font_dataset")
```

**2. Save to persistent storage**:
```python
# In Colab, save to Drive:
save_dir = "/content/drive/MyDrive/models"

# Or download after training:
from google.colab import files
files.download('models/best_model.pt')
```

**3. Monitor progress**:
```python
# Colab shows output in real-time
# Kaggle shows in notebook output
```

---

## Cost Estimates

### Free Options:
- **Colab**: $0 (free tier) or $10-20/month (Pro)
- **Kaggle**: $0 (30 hours/week)

### Paid Options (for 10 epochs, ~1 hour):
- **GCP T4**: ~$0.35
- **AWS g4dn**: ~$0.53
- **Paperspace**: ~$0.51

### For Full Training (20 epochs, ~2 hours):
- **GCP**: ~$0.70
- **AWS**: ~$1.06
- **Paperspace**: ~$1.02

**Very affordable!** Even paid options cost less than $2 for full training.

---

## Step-by-Step: Using Google Colab

### 1. Prepare Your Project

**Option A: Upload to Google Drive**
```
1. Zip your check_fonts folder
2. Upload to Google Drive
3. Unzip in Colab
```

**Option B: Use GitHub**
```
1. Push code to GitHub
2. Clone in Colab: !git clone https://github.com/yourusername/check_fonts.git
```

### 2. Create Colab Notebook

```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install dependencies
!pip install torch torchvision open-clip-torch tqdm pillow numpy

# Cell 3: Navigate to project
%cd /content/drive/MyDrive/check_fonts

# Cell 4: Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 5: Run training
!python train_embedding_model.py \
  --epochs 10 \
  --batch_size 32 \
  --dataset_dir /content/drive/MyDrive/check_fonts/font_dataset \
  --save_dir /content/drive/MyDrive/check_fonts/models
```

### 3. Monitor Training

- Watch output in notebook cells
- Check GPU usage: Runtime → Manage sessions
- Download model when done

---

## Tips for Cloud GPU Usage

### 1. Save Frequently
- Save checkpoints to Google Drive
- Download important files before session ends

### 2. Use Persistent Storage
- Mount Google Drive (Colab)
- Use Kaggle datasets (Kaggle)

### 3. Monitor Usage
- Colab: Check session time remaining
- Kaggle: Check weekly hours used

### 4. Optimize for Cloud
- Use larger batch sizes (more GPU memory available)
- Enable mixed precision training (faster)

---

## Quick Start Commands

### Google Colab
```python
# Full setup in one cell
!pip install torch torchvision open-clip-torch
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/check_fonts
!python train_embedding_model.py --epochs 10 --batch_size 64
```

### Kaggle
```python
# In Kaggle notebook
!pip install open-clip-torch
!python train_embedding_model.py --epochs 10 --batch_size 64
```

---

## FAQ

**Q: Which is best for my MVP?**  
A: **Google Colab** - easiest, free, perfect for testing

**Q: How much will it cost?**  
A: **$0** with free options, or ~$1-2 for paid cloud GPUs

**Q: Will training be faster on cloud GPU?**  
A: Yes! Cloud GPUs are often faster than CPU training by 10-50×

**Q: Can I use my local code?**  
A: Yes! Just upload to Drive/GitHub and run in cloud

**Q: What if session times out?**  
A: Save checkpoints! You can resume training from last checkpoint

**Q: Is my data safe?**  
A: Yes, but don't upload sensitive data. Use Google Drive for privacy.

---

## Recommendation

**For your MVP**: Start with **Google Colab** (free, easy)

**Steps**:
1. ✅ Test your code locally first (CPU, 1 epoch)
2. ✅ Upload to Google Drive
3. ✅ Run in Colab with GPU
4. ✅ Download trained model

**Time saved**: Instead of 2-4 hours on CPU, train in 30-60 minutes on GPU!

---

## Next Steps

1. **Try Google Colab** (free, easiest)
2. **If you need more time**: Consider Kaggle (30h/week)
3. **If you need production**: Use paid cloud (GCP/AWS)

**You don't need to buy a GPU to train your model!** 🚀

