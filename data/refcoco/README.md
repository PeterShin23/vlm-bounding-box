# RefCOCO Local Data

This directory contains exported RefCOCO samples for easy viewing and offline access.

## ⚠️ IMPORTANT: Split Naming

RefCOCO has a confusing split structure on HuggingFace:

- **`val/`** → **TRAINING DATA** (8,811 samples total, 1,000 exported here)
- **`test/`** → **EVALUATION DATA** (5,000 samples total, 500 exported here)

**There is NO "train" split!** Use the `val` split for training.

## Directory Structure

```
refcoco/
├── val/                    # TRAINING data
│   ├── images/            # Images (val_00000.jpg, val_00001.jpg, ...)
│   └── metadata.json      # Phrases and bounding boxes
└── test/                   # EVALUATION data
    ├── images/            # Images (test_00000.jpg, test_00001.jpg, ...)
    └── metadata.json      # Phrases and bounding boxes
```

## Metadata Format

Each `metadata.json` contains:

```json
{
  "split": "val",
  "variant": "refcoco",
  "num_samples": 1000,
  "samples": [
    {
      "id": 0,
      "image_path": "images/val_00000.jpg",
      "width": 640,
      "height": 428,
      "phrase": "the person on the right",
      "bbox_norm": {
        "x_min": 0.73,
        "y_min": 0.00,
        "x_max": 1.00,
        "y_max": 0.27
      }
    }
  ]
}
```

## Viewing the Data

**Browse images in Finder:**
```bash
open val/images/
open test/images/
```

**View metadata:**
```bash
cat val/metadata.json | python3 -m json.tool | less
```

## Export More Data

To export additional samples:

```bash
# Export all val samples (8,811 images, ~1.1 GB)
python scripts/export_refcoco.py --split val

# Export all test samples (5,000 images, ~680 MB)
python scripts/export_refcoco.py --split test
```
