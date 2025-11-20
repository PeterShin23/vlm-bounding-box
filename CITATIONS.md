# Dataset Citations

This project uses the **DUTS (Densely Annotated UTterance Saliency)** dataset for training and evaluation.

## DUTS Dataset

### Citation

**Paper:**
Lijun Wang, Huchuan Lu, Yifan Wang, Mengyang Feng, Dong Wang, Baocai Yin, Xiang Ruan. "Learning to Detect Salient Objects with Image-level Supervision", CVPR 2017.

**BibTeX:**
```bibtex
@inproceedings{wang2017duts,
  title={Learning to Detect Salient Objects with Image-level Supervision},
  author={Wang, Lijun and Lu, Huchuan and Wang, Yifan and Feng, Mengyang and Wang, Dong and Yin, Baocai and Ruan, Xiang},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```

### Dataset Information

**Official Website:** http://saliencydetection.net/duts/

**Dataset Statistics:**
- **DUTS-TR (Training Set)**: 10,553 images with pixel-level annotations
- **DUTS-TE (Test Set)**: 5,019 images with pixel-level annotations

**Data Sources:**
- Training images: ImageNet DET training/val sets
- Test images: ImageNet DET test set and SUN dataset

**Annotations:**
- Pixel-level ground truth masks
- Manually annotated by 50 subjects
- Challenging scenarios for saliency detection

### Copyright and Usage

**Copyright:** All rights reserved by the original authors of the DUTS Image Dataset.

**Usage:** When using this dataset, please cite the original CVPR 2017 paper above.

### Important Dataset Update (2018-01-22)

The dataset was updated in January 2018 to correct some errors. If you downloaded the dataset before this date, please re-download or apply the following corrections manually:

**DUTS-TE corrections:**
- Delete files: `ILSVRC2012_test_00036002.jpg`, `sun_bcogaqperiljqupq.jpg`

**DUTS-TR corrections:**
- Delete files: `ILSVRC2014_train_00023530.png`, `n01532829_13482.png`, `n04442312_17818.png`
- Convert to PNG and remove JPG: `ILSVRC2014_train_00023530.jpg`, `n01532829_13482.jpg`, `n04442312_17818.jpg`

The download links provided in this project point to the corrected version (post-2018).

## Dataset Description

### Purpose

DUTS was created to address limitations in existing saliency detection datasets:
- Previous datasets had insufficient samples for training deep neural networks
- No well-established train/test protocol for fair comparison
- Need for more challenging and diverse scenarios

### Key Features

1. **Large Scale**: Currently the largest saliency detection benchmark with explicit train/test split
2. **High Quality**: Pixel-level annotations with careful manual verification
3. **Challenging**: Contains complex scenes with multiple objects, cluttered backgrounds
4. **Standardized**: Provides consistent training/test protocol for fair method comparison

### Recommended Usage

For this project:
- Use **DUTS-TR** as the source for train/val/test splits
- Default split: 2,000 train / 500 val / 500 test (configurable)
- Remaining images can be used for additional experiments

## Alternative Datasets

If DUTS is unavailable, the following alternatives can be used:

### MSRA-B
- ~2,500 images
- Note: Some official download links may be unavailable
- Smaller but sufficient for initial experiments

### DUT-OMRON
- ~5,000 images
- Website: http://saliencydetection.net/dut-omron/
- Good alternative with moderate size

## Attribution Requirements

When publishing work using this project and the DUTS dataset:

1. **Cite the DUTS paper** (see BibTeX above)
2. **Cite this project** (if using the code):
   ```bibtex
   @misc{qwen3-vl-bbox,
     title={Qwen3-VL Main Subject Bounding Box Detection},
     author={Your Name},
     year={2025},
     howpublished={\url{https://github.com/yourusername/qwen3-vl-bbox}}
   }
   ```

## Questions or Issues

For questions about the DUTS dataset itself, please contact the original authors or visit:
http://saliencydetection.net/duts/

For questions about using DUTS with this project, please open an issue in this repository.

---

**Last Updated:** 2025-01-19
