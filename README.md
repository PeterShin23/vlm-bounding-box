# Qwen3-VL Phrase Grounding (RefCOCO)

A focused multimodal fine-tuning project that fine-tunes **Qwen3-VL-2B** to predict JSON bounding boxes for objects described by natural language phrases.

## Demo Notebook

You can view the full training/evaluations/examples here:

[NBViewer link](https://nbviewer.org/github/PeterShin23/vlm-bounding-box/blob/main/notebooks/train_qwen3_refcoco.ipynb)

## Project Overview

This project demonstrates end-to-end VLM (Vision-Language Model) fine-tuning using:

- **Model**: Qwen3-VL-2B-Instruct
- **Fine-tuning method**: LoRA/QLoRA for efficient training
- **Task**: Referring expression grounding (phrase-conditional bounding box prediction)
- **Dataset**: RefCOCO from HuggingFace (`lmms-lab/RefCOCO`)
- **Platform**: Mac M-series (M4 Max, 36GB) with MPS for debugging + Runpod GPU for training

### Learning Goals

- End-to-end LLM/VLM fine-tuning with LoRA
- Multimodal prompting (image + phrase → bounding box)
- Referring expression grounding
- Evaluation metrics for spatial predictions (IoU)
- Budget-aware cloud GPU training
- Production-ready project structure
