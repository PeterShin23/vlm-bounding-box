# Outputs Directory

This folder centralizes artifacts produced while training and evaluating the RefCOCO grounding models. Everything is organized so it is easy to locate checkpoints or diagnostics later.

```
outputs/
├── checkpoints/          # Model weights + training snapshots
│   └── qwen3_refcoco/
│       ├── epochs/       # LoRA checkpoints saved at the end of each epoch
│       ├── steps/        # Intermediate steps (e.g., step_200 warm-start)
│       ├── final/        # Final merged weights for deployment
│       ├── final_lora_weights/  # Adapter-only weights for PEFT loading
│       └── archive/      # Timestamped raw checkpoints kept for provenance
├── evaluations/          # (empty placeholder) stores JSON/CSV metric dumps
├── logs/                 # Training metadata (loss curves, histories, etc.)
└── visualizations/       # PNGs/JPEGs rendered from evaluation notebooks
```

### Notes

- Keep only the checkpoints you need for reproducibility. Older raw checkpoints are tucked under `checkpoints/qwen3_refcoco/archive/` to avoid clutter but remain available.
- Place future evaluation exports (e.g., IoU summaries, qualitative tables) under `outputs/evaluations/` so they do not get lost among checkpoints.
- Use `outputs/logs/` for serialized histories (`training_history.json`, TensorBoard exports, etc.) to separate them from the live `logs/` folder at the repo root (which now holds inference attempt CSVs).
