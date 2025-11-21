# Migration Guide: DUTS Saliency → RefCOCO Phrase Grounding

This document tracks the pivot from saliency-based main subject detection to RefCOCO referring expression grounding.

## ✅ Changes Completed

### 1. Core Data Layer
- **`src/data/box_utils.py`**: Added `coco_to_xyxy()` function for COCO format conversion
- **`src/data/refcoco_dataset.py`**: New dataset class for RefCOCO with HuggingFace integration
  - Supports `max_samples` for budget-aware training
  - Handles (image, phrase, bbox) tuples
  - Robust field name detection for different HF dataset variants

### 2. Prompting System
- **`src/pipeline/prompts.py`**: Completely rewritten for phrase-conditional grounding
  - `build_grounding_prompt(phrase)` - takes referring expression as input
  - Updated message formatting to include phrase in prompt

### 3. Device Management
- **`src/common/device.py`**: Added Runpod support
  - `get_device(prefer_cuda=False)` - smart device selection
  - `prefer_cuda=True` → prioritizes CUDA (for Runpod)
  - `prefer_cuda=False` → prioritizes MPS (for local Mac)
  - Added `print_device_info()` for debugging

## 🔧 Changes Needed

### 1. Pipeline Updates (CRITICAL)

**`src/pipeline/training.py`**:
```python
# OLD (line ~140):
prompt = build_main_subject_prompt()

# NEW:
phrases = batch["phrase"]  # Extract phrases from batch
prompts = [build_grounding_prompt(phrase) for phrase in phrases]
```

Key changes needed:
- Extract `phrase` from batch alongside image
- Build per-sample prompts using `build_grounding_prompt(phrase)`
- Rest of training loop stays the same

**`src/pipeline/eval.py`**:
```python
# OLD:
predict_bbox(image, ...)

# NEW:
predict_bbox(image, phrase, ...)
```

Changes:
- Add `phrase` parameter to prediction function
- Pass phrase to `build_grounding_prompt()` during evaluation

**`src/pipeline/inference.py`**:
```python
# OLD signature:
def predict_main_subject_bbox(image, model, processor, device, ...)

# NEW signature:
def predict_bbox_for_phrase(image, phrase, model, processor, device, ...)
```

Changes:
- Add `phrase` as required parameter
- Use `build_grounding_prompt(phrase)` instead of fixed prompt

### 2. Data Preparation Script

**`scripts/prepare_data.py`** - needs complete rewrite for RefCOCO:

```python
#!/usr/bin/env python3
"""
RefCOCO data preparation and verification script.
"""
from datasets import load_dataset
from src.data.refcoco_dataset import RefCOCODataset
from src.common.viz import draw_bbox_on_image, create_grid

def main():
    # Load RefCOCO from HuggingFace
    print("Loading RefCOCO dataset from HuggingFace...")
    dataset = load_dataset("lmms-lab/RefCOCO", "refcoco", trust_remote_code=True)

    # Print dataset info
    print(f"\nDataset splits: {dataset.keys()}")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} samples")

    # Create small subsets for local testing
    # (These can be saved as separate files if needed)

    # Visualize samples
    print("\nCreating visualizations...")
    # ... (similar to old prepare_data.py but for RefCOCO)

if __name__ == "__main__":
    main()
```

### 3. Configuration Updates

**`configs/data.yaml`**:
```yaml
# RefCOCO dataset configuration
dataset:
  name: "lmms-lab/RefCOCO"
  variant: "refcoco"  # or "refcoco+", "refcocog"

  # Budget-aware sampling
  max_train_samples: null  # null = use all, or set to 5000, 10000, etc.
  max_val_samples: 2000
  max_test_samples: 2000

# DataLoader settings
dataloader:
  batch_size: 4  # Local MPS
  num_workers: 0
  pin_memory: false

# Runpod overrides (use larger batch with grad accumulation)
runpod:
  batch_size: 2
  gradient_accumulation_steps: 4  # Effective batch size = 8
```

**`configs/training.yaml`**:
```yaml
# Training configuration
training:
  num_epochs: 3
  learning_rate: 2.0e-4
  max_steps: null  # Or set for budget control (e.g., 5000)

  # Budget-aware settings
  eval_steps: 500  # Evaluate every N steps
  save_steps: 1000  # Save every N steps

# Device selection
device:
  prefer_cuda: false  # Set to true for Runpod

# Runpod budget config
runpod_budget:
  max_train_samples: 10000  # ~$5-8 on mid-tier GPU
  num_epochs: 2
  max_steps: 5000  # Early stopping
```

### 4. Training Scripts

**`scripts/train_local.sh`** (new file):
```bash
#!/bin/bash
# Quick local debugging run on MPS

python scripts/train.py \
    --config configs/training.yaml \
    --data_config configs/data.yaml \
    --max_train_samples 500 \
    --max_val_samples 100 \
    --num_epochs 1 \
    --batch_size 2 \
    --device mps
```

**`scripts/train_runpod.sh`** (new file):
```bash
#!/bin/bash
# Budget-conscious Runpod training

python scripts/train.py \
    --config configs/training.yaml \
    --data_config configs/data.yaml \
    --max_train_samples 10000 \
    --max_val_samples 2000 \
    --num_epochs 2 \
    --max_steps 5000 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --prefer_cuda \
    --output_dir /workspace/outputs
```

### 5. Scripts Update

**`scripts/train.py`** - add CLI arguments:
```python
parser.add_argument("--prefer_cuda", action="store_true", help="Prefer CUDA over MPS")
parser.add_argument("--max_train_samples", type=int, help="Limit training samples")
parser.add_argument("--max_val_samples", type=int, help="Limit validation samples")
parser.add_argument("--max_steps", type=int, help="Maximum training steps (for budget)")

# Then use:
device = get_device(prefer_cuda=args.prefer_cuda)
```

## 📊 Updated Task Description

**Old Task**: Given an image, predict a bounding box around the main salient subject.
- Input: Image only
- Output: JSON bbox
- Dataset: DUTS saliency masks

**New Task**: Given an image and a referring expression, predict the bounding box around the described region.
- Input: Image + phrase (text)
- Output: JSON bbox
- Dataset: RefCOCO (COCO images + referring expressions)

## 🎯 Budget Strategy

### Local (MPS) - Free
- Quick debugging: 500-1000 samples, 1 epoch (~10-20 min)
- Purpose: Verify code works, debug issues

### Runpod - Under $10 Total

**Option 1: Conservative** (~$3-5)
- GPU: L4 or A10G (~$0.30-0.50/hr)
- Training: 5k samples, 2 epochs
- Time: ~3-6 hours
- Cost: ~$1.50-3.00

**Option 2: Moderate** (~$6-9)
- GPU: RTX 3090 or A10G (~$0.50-0.70/hr)
- Training: 10-15k samples, 2-3 epochs
- Time: ~8-12 hours
- Cost: ~$4-8

**Budget Control Mechanisms**:
1. `max_train_samples` - limit dataset size
2. `max_steps` - early stopping
3. `eval_steps` - monitor and stop manually if needed

## 🔄 Step-by-Step Migration Checklist

- [x] Update `box_utils.py` with COCO conversion
- [x] Create `refcoco_dataset.py`
- [x] Rewrite `prompts.py` for phrase input
- [x] Update `device.py` for Runpod
- [ ] Update `training.py` batch preparation (add phrase extraction)
- [ ] Update `eval.py` prediction (add phrase parameter)
- [ ] Update `inference.py` (add phrase parameter)
- [ ] Rewrite `prepare_data.py` for RefCOCO
- [ ] Update `configs/data.yaml`
- [ ] Update `configs/training.yaml`
- [ ] Create `scripts/train_local.sh`
- [ ] Create `scripts/train_runpod.sh`
- [ ] Update `scripts/train.py` CLI args
- [ ] Update `README.md` with new task description
- [ ] Update `CITATIONS.md` with RefCOCO citation

## 📖 RefCOCO Citation

Add to `CITATIONS.md`:

```bibtex
@inproceedings{yu2016refcoco,
  title={Modeling Context in Referring Expressions},
  author={Yu, Licheng and Poirson, Patrick and Yang, Shan and Berg, Alexander C and Berg, Tamara L},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2016}
}
```

## 🚀 Quick Start After Migration

1. **Test dataset loading**:
```python
from src.data.refcoco_dataset import RefCOCODataset
dataset = RefCOCODataset(split="train", max_samples=10)
sample = dataset[0]
print(f"Phrase: {sample['phrase']}")
print(f"BBox: {sample['bbox_norm']}")
```

2. **Local debug run**:
```bash
bash scripts/train_local.sh
```

3. **Runpod training**:
```bash
# On Runpod container:
bash scripts/train_runpod.sh
```

## 💡 Key Differences Summary

| Aspect | Old (DUTS Saliency) | New (RefCOCO Grounding) |
|--------|---------------------|-------------------------|
| Task | Main subject detection | Phrase grounding |
| Input | Image only | Image + phrase |
| Dataset | DUTS (10k images) | RefCOCO (HF dataset) |
| Prompt | Fixed template | Phrase-conditional |
| Device | MPS only | MPS (local) + CUDA (Runpod) |
| Training | Local only | Local debug + Cloud training |
| Budget | Free | ~$10 cloud budget |

