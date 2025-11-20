# Qwen3-VL Main Subject Bounding Box Detection

A focused multimodal fine-tuning project that fine-tunes **Qwen3-VL-2B** to predict JSON bounding boxes around the main subject in images.

## Project Overview

This project demonstrates end-to-end VLM (Vision-Language Model) fine-tuning using:

- **Model**: Qwen3-VL-2B-Instruct
- **Fine-tuning method**: LoRA/QLoRA for efficient training
- **Task**: Predict normalized bounding box coordinates (JSON) for the main salient subject
- **Dataset**: DUTS saliency detection dataset (recommended) or alternatives (MSRA-B, DUT-OMRON)
- **Platform**: Mac M-series (M4 Max, 36GB) with MPS acceleration

### Learning Goals

- End-to-end LLM/VLM fine-tuning with LoRA
- Multimodal prompting (image + instruction → text)
- Evaluation metrics for spatial predictions (IoU)
- Production-ready project structure

## Repository Structure

```
vlm-bounding-box/
├── configs/
│   ├── data.yaml          # Dataset configuration
│   └── training.yaml      # Training hyperparameters
├── src/
│   ├── common/
│   │   ├── device.py      # MPS/CPU device selection
│   │   ├── paths.py       # Path management
│   │   └── viz.py         # Visualization utilities
│   ├── data/
│   │   ├── box_utils.py   # Bbox operations, IoU, JSON conversion
│   │   ├── saliency_dataset.py  # PyTorch dataset
│   │   └── split_utils.py       # Train/val/test splitting
│   └── pipeline/
│       ├── prompts.py     # Instruction templates
│       ├── model_qwen3.py # Model loading + LoRA setup
│       ├── training.py    # Training loop
│       ├── eval.py        # Evaluation with IoU metrics
│       └── inference.py   # Single-image prediction
├── scripts/
│   ├── prepare_data.py    # Dataset preparation
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── inference.py       # Inference script
├── data/
│   ├── raw/               # Raw datasets
│   └── processed/         # Processed splits
├── outputs/
│   ├── checkpoints/       # Model checkpoints
│   ├── logs/              # Training logs
│   └── visualizations/    # Sample visualizations
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
cd vlm-bounding-box
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Important for Mac M-series users:**

- The code is optimized for MPS (Metal Performance Shaders)
- Quantization (bitsandbytes) may not work well on MPS - use full precision instead
- Set `use_quantization: false` in `configs/training.yaml`

## Quick Start

### Step 1: Download Dataset

Get dataset download instructions:

```bash
python3 scripts/prepare_data.py --download_help
```

Recommended dataset:

- **DUTS (recommended)**: 10,553 training images, high quality, actively maintained
  - Download: http://saliencydetection.net/duts/
  - Direct link: http://saliencydetection.net/duts/download/DUTS-TR.zip (~800MB)
  - See [CITATIONS.md](CITATIONS.md) for proper attribution

Alternative datasets:
- **MSRA-B**: ~2,500 images (note: some download links may be broken)
- **DUT-OMRON**: ~5,000 images

After downloading and extracting, organize to get this structure:

```
/path/to/dataset/
  images/
    img1.jpg
    img2.jpg
    ...
  masks/
    img1.png
    img2.png
    ...
```

### Step 2: Prepare Data

Process the raw dataset and create train/val/test splits:

```bash
python3 scripts/prepare_data.py \
    --data_dir /path/to/dataset \
    --output_dir data/processed \
    --train_size 2000 \
    --val_size 500 \
    --test_size 500 \
    --visualize
```

This will:

- Compute bounding boxes from saliency masks
- Create train/val/test splits
- Save processed data to `data/processed/`
- Generate sample visualizations in `outputs/visualizations/`

### Step 3: Train the Model

Start fine-tuning:

```bash
python3 scripts/train.py --config configs/training.yaml
```

For quick debugging with a small subset:

```bash
python3 scripts/train.py --debug
```

Training outputs:

- Checkpoints: `outputs/checkpoints/`
- Logs: `outputs/logs/training_history.json`

### Step 4: Evaluate

Evaluate the trained model on the test set:

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/final \
    --split test
```

This will report:

- Mean IoU (Intersection over Union)
- Success rate at IoU ≥ 0.5
- Success rate at IoU ≥ 0.75
- Parsing success rate

### Step 5: Run Inference

Predict bounding box on a single image:

```bash
python3 scripts/inference.py \
    --image /path/to/image.jpg \
    --checkpoint outputs/checkpoints/final \
    --output outputs/predictions/result.png
```

## 5-Day Implementation Plan

This project is designed to be completed in approximately 5 days of focused work:

### Day 1: Data Preparation

- Download DUTS saliency dataset (10K+ images, recommended)
- Run `prepare_data.py` to process and create train/val/test splits
- Verify splits and visualizations
- Understand the data format and bounding box computation

**Key files**: `src/data/box_utils.py`, `src/data/saliency_dataset.py`

### Day 2: Model Setup & Verification

- Install dependencies and verify environment
- Load Qwen3-VL-2B with LoRA configuration
- Test a single forward pass (no training)
- Understand the prompt format and tokenization

**Key files**: `src/pipeline/model_qwen3.py`, `src/pipeline/prompts.py`

### Day 3: Initial Training

- Run training on a small subset (~100 images) to debug
- Monitor loss curves and sample outputs
- Fix any issues with batch preparation or LoRA setup
- Adjust hyperparameters if needed

**Key files**: `src/pipeline/training.py`

### Day 4: Full Training

- Train on full dataset (~2000 train, 500 val)
- Monitor training loss and validation IoU
- Let training run overnight if needed
- Save checkpoints at each epoch

**Expected**: 3 epochs, ~6-12 hours on M4 Max

### Day 5: Evaluation & Testing

- Evaluate on test set with IoU metrics
- Generate visualizations of predictions
- Test inference on new images
- Document results and learnings

**Key files**: `src/pipeline/eval.py`, `src/pipeline/inference.py`

## Configuration

### Data Configuration (`configs/data.yaml`)

Key parameters:

- `splits`: train/val/test sizes
- `processing.resize`: Image resize (e.g., 320px)
- `dataloader.batch_size`: Batch size (4 recommended)

### Training Configuration (`configs/training.yaml`)

Key parameters:

- `model.name`: Base model (default: Qwen/Qwen3-VL-2B-Instruct)
- `lora.r`: LoRA rank (16 = good balance)
- `training.learning_rate`: Learning rate (2e-4 recommended)
- `training.num_epochs`: Number of epochs (3 recommended)

For Mac M-series:

- Set `use_quantization: false`
- Set `device.type: "mps"`
- Keep `num_workers: 0` in dataloader

## How It Works

### Task Definition

**Input**:

- An RGB image
- Fixed instruction: "Find the main subject and output a JSON bounding box"

**Output**:

```json
{
  "x_min": 0.13,
  "y_min": 0.22,
  "x_max": 0.54,
  "y_max": 0.78
}
```

All coordinates are normalized to [0, 1].

### Training Process

1. **Data**: Load image + saliency mask
2. **Ground Truth**: Compute bounding box from mask (largest connected component)
3. **Prompt**: Format as Qwen3-VL chat message with image + instruction
4. **Target**: JSON string of normalized coordinates
5. **Loss**: Standard next-token cross-entropy on JSON tokens only (prompt masked with -100)
6. **Optimization**: LoRA adapters updated, base model frozen

### Evaluation Metrics

- **Mean IoU**: Average Intersection over Union with ground truth
- **Success@0.5**: Percentage of predictions with IoU ≥ 0.5
- **Success@0.75**: Percentage of predictions with IoU ≥ 0.75
- **Parse Rate**: Percentage of valid JSON outputs

## Advanced Usage

### Custom Dataset

To use your own dataset:

1. Organize as `images/` and `masks/` folders
2. Update `configs/data.yaml` with paths and file extensions
3. Run `prepare_data.py` with your dataset

### Hyperparameter Tuning

Key hyperparameters to experiment with:

- **LoRA rank (`lora.r`)**: Higher = more capacity but slower

  - 8: Fast, good for simple tasks
  - 16: Balanced (default)
  - 32: More capacity, slower

- **Learning rate**:

  - Too high: Unstable training
  - Too low: Slow convergence
  - Recommended: 1e-4 to 5e-4

- **Batch size**: Limited by memory
  - Use `gradient_accumulation_steps` to simulate larger batches

### Visualization

Generate visualizations during evaluation:

```bash
python3 scripts/evaluate.py \
    --checkpoint outputs/checkpoints/final \
    --split test
```

Creates comparison images with GT (green) and prediction (red) boxes.

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in `configs/data.yaml`
- Increase `gradient_accumulation_steps`
- Reduce `processing.resize` image size
- Use smaller LoRA rank

### Poor IoU Scores

- Check data quality (visualize samples)
- Train for more epochs
- Increase LoRA rank
- Adjust learning rate
- Ensure masks properly represent main subjects

### Model Not Loading

- Check internet connection (downloads from HuggingFace)
- Verify transformers version: `pip install transformers>=4.37.0`
- For Qwen models, may need: `pip install transformers_stream_generator tiktoken`

### MPS Errors on Mac

- Some operations may not be supported on MPS
- Fallback to CPU if needed: set `device.type: "cpu"` in config
- Update PyTorch: `pip install --upgrade torch torchvision`

## Results Interpretation

After training, you should expect:

**Good results**:

- Mean IoU: 0.6 - 0.8
- Success@0.5: 70% - 90%
- Parse rate: > 95%

**If results are poor**:

1. Check data quality (are masks accurate?)
2. Visualize predictions to identify failure modes
3. Train longer or with larger LoRA rank
4. Ensure training loss is decreasing

## Project Philosophy

This project prioritizes:

✅ **Clarity over optimization**: Code is readable and well-documented
✅ **Completeness**: Full pipeline from data to evaluation
✅ **Reproducibility**: Fixed seeds, clear configs
✅ **Learning**: Extensive comments explaining VLM fine-tuning

Not prioritized:
❌ Bleeding-edge performance
❌ Production deployment features
❌ Multi-GPU training

## Future Improvements

Potential extensions (not implemented):

- Multiple objects per image
- Category-specific detection
- Confidence scores
- Integration with object detection models
- Web demo interface

## Citation

If you use this code for your projects or research:

```bibtex
@misc{qwen3-vl-bbox,
  title={Qwen3-VL Main Subject Bounding Box Detection},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/qwen3-vl-bbox}}
}
```

## References

### Models and Libraries
- **Qwen3-VL**: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- **LoRA Paper**: "LoRA: Low-Rank Adaptation of Large Language Models"
- **PEFT Library**: https://github.com/huggingface/peft

### Datasets
- **DUTS Dataset**: Lijun Wang, Huchuan Lu, Yifan Wang, Mengyang Feng, Dong Wang, Baocai Yin, Xiang Ruan. "Learning to Detect Salient Objects with Image-level Supervision", CVPR 2017.
  - Website: http://saliencydetection.net/duts/
  - See [CITATIONS.md](CITATIONS.md) for full BibTeX citation and copyright information

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Qwen team for the excellent VLM models
- HuggingFace for transformers and PEFT libraries
- DUTS dataset authors (Wang et al., CVPR 2017) for the high-quality saliency dataset
- All saliency detection dataset creators (MSRA, DUT, DUTS, etc.)

---

**Happy fine-tuning! 🚀**

For questions or issues, please open a GitHub issue or check the troubleshooting section.
