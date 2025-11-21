# Data Directory

## RefCOCO Dataset

This project uses the **RefCOCO** dataset for referring expression grounding. The dataset is available both:
1. **Locally** in this directory (for easy viewing and offline use)
2. **From HuggingFace** cache (automatic fallback)

### Local RefCOCO Data (In This Repo)

The RefCOCO data is stored locally in this directory for easy browsing:

```
data/refcoco/
├── val/                    # Training data (1,000 samples exported)
│   ├── images/            # 1,000 images (131 MB)
│   │   ├── val_00000.jpg
│   │   ├── val_00001.jpg
│   │   └── ...
│   └── metadata.json      # Phrases, bboxes, image info
└── test/                   # Evaluation data (500 samples exported)
    ├── images/            # 500 images (68 MB)
    │   ├── test_00000.jpg
    │   ├── test_00001.jpg
    │   └── ...
    └── metadata.json      # Phrases, bboxes, image info
```

**Total exported**: 1,500 images (~200 MB)

### Viewing the Data

**Browse images:**
```bash
open data/refcoco/val/images/    # View training images
open data/refcoco/test/images/   # View evaluation images
```

**View metadata (phrases and bboxes):**
```bash
cat data/refcoco/val/metadata.json | jq '.samples[0]'
```

Example metadata entry:
```json
{
  "id": 0,
  "image_path": "images/val_00000.jpg",
  "width": 640,
  "height": 428,
  "phrase": "Please carefully observe the area circled...",
  "bbox_norm": {
    "x_min": 0.7317,
    "y_min": 0.0021,
    "x_max": 1.0,
    "y_max": 0.2734
  }
}
```

### Exporting More Data

To export additional samples or full splits:

```bash
# Export more val samples (e.g., 5000 samples)
python scripts/export_refcoco.py --split val --max_samples 5000

# Export all test samples
python scripts/export_refcoco.py --split test

# Export everything (all val + all test = 13,811 images, ~1.8 GB)
python scripts/export_refcoco.py --all
```

### Full Dataset Splits

The complete RefCOCO dataset (in HuggingFace cache):
- **val**: 8,811 samples (use for **training**)
- **test**: 5,000 samples (use for **evaluation**)
- **testA**: 1,975 samples (people subset)
- **testB**: 1,810 samples (objects subset)

⚠️ **IMPORTANT**: RefCOCO has NO "train" split - use "val" for training!

### HuggingFace Cache

The full dataset is also cached automatically:
- **Location**: `~/.cache/huggingface/datasets/lmms-lab___ref_coco/`
- **Size**: ~2.9 GB (all images + metadata)
- **Auto-managed**: Downloaded on first use

### First-Time Setup

1. Ensure you have the `datasets` library installed:
   ```bash
   pip install datasets>=2.14.0
   ```

2. Run the data preparation script:
   ```bash
   python scripts/prepare_data.py --split val --num_samples 5 --visualize
   ```

3. This will:
   - Download RefCOCO from HuggingFace (first time only)
   - Cache it in `~/.cache/huggingface/`
   - Generate visualizations in `outputs/visualizations/`

### Offline Usage

If you need to use RefCOCO offline:

1. Download once while online:
   ```bash
   python scripts/prepare_data.py --test
   ```

2. HuggingFace will cache all data locally

3. Future runs will use the cache (no internet needed)

### Clearing Cache

If you need to re-download or clear space:

```bash
# Remove all HuggingFace datasets cache
rm -rf ~/.cache/huggingface/datasets/

# Remove just RefCOCO
rm -rf ~/.cache/huggingface/datasets/lmms-lab___ref_coco/
```

### Migration from DUTS

This project was originally designed for DUTS saliency dataset but has been migrated to RefCOCO for referring expression grounding. The old DUTS data has been removed. See `MIGRATION_REFCOCO.md` for details.

---

**Last Updated**: 2025-01-20
