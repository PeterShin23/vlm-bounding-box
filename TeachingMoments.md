# Teaching Moments: PyTorch Mastery

This document serves as a living knowledge base - a collection of concepts, techniques, and insights encountered while working on this project. Each entry captures a learning moment, providing deep explanations that build toward PyTorch mastery.

---

## Table of Contents

1. [Model Quantization & Bitsandbytes](#1-model-quantization--bitsandbytes)
2. [Label Masking in Vision-Language Model Training](#2-label-masking-in-vision-language-model-training)
3. [Interpreting Detection Metrics & Improving Grounding Models](#3-interpreting-detection-metrics--improving-grounding-models)

---

## 1. Model Quantization & Bitsandbytes

**Date:** 2025-11-21
**Context:** Loading Qwen3-VL model with `use_quantization=False` parameter

### What is Quantization?

Quantization is the process of converting a neural network's weights and activations from high-precision numbers (typically 32-bit floating point) to lower-precision formats (8-bit or 4-bit integers).

**The basics:**

```python
# Normal (full precision)
weight = 0.123456789  # 32-bit float (FP32) - 4 bytes per number

# Quantized (8-bit integer)
weight = 127  # 8-bit int - 1 byte per number
# Maps to range [-1.0, 1.0] approximately
```

**Why does this matter?**

A 2B parameter model like Qwen3-VL:
- **FP32 (full precision):** 2B parameters × 4 bytes = ~8 GB
- **INT8 (8-bit quantization):** 2B parameters × 1 byte = ~2 GB
- **INT4 (4-bit quantization):** 2B parameters × 0.5 bytes = ~1 GB

You get 4-8x memory savings!

### Types of Quantization

1. **Post-Training Quantization (PTQ)**
   - Quantize a trained model
   - Fast, no retraining needed
   - Small accuracy loss (~1-2%)

2. **Quantization-Aware Training (QAT)**
   - Train model knowing it will be quantized
   - Better accuracy preservation
   - More expensive (full training)

### What is Bitsandbytes?

**Bitsandbytes** is a library that implements efficient quantization algorithms for PyTorch models. It's particularly popular for:

1. **8-bit optimizers** - Reduce optimizer memory (AdamW stores 2 extra copies of parameters!)
2. **4-bit/8-bit model loading** - Load huge models in less memory
3. **QLoRA** - Train quantized models with LoRA adapters

**How it works:**

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16, # Compute in FP16 for accuracy
    bnb_4bit_use_double_quant=True,       # Quantize the quantization constants
    bnb_4bit_quant_type="nf4"            # Use NormalFloat4 (optimized for neural nets)
)

model = AutoModel.from_pretrained(
    "model_name",
    quantization_config=quantization_config
)
```

When you load a model with this config, bitsandbytes:
1. Loads weights from disk
2. Converts FP32 → INT4 on-the-fly
3. Stores them in ~1 GB instead of ~8 GB
4. During forward pass: INT4 → FP16 for computation → INT4 for storage

### The `use_quantization=False` Parameter

In our project (`src/pipeline/model_qwen3.py:26`), we have:

```python
def load_qwen3_vl_with_lora(
    use_quantization: bool = True,
    quantization_bits: int = 4,
    ...
):
```

**Why is it set to `False` in the notebook?**

```python
# From model_qwen3.py:88-93
if use_quantization:
    if device_str == "mps":
        print("Warning: Quantization not well supported on MPS. Using full precision.")
        use_quantization = False
```

**The reason:** Bitsandbytes was designed for NVIDIA GPUs (CUDA) and doesn't support:
- Mac's MPS (Metal Performance Shaders) backend
- CPU-only training

**What happens on Mac:**
- Code detects MPS device
- Automatically disables quantization
- Falls back to full precision (FP32)
- Model uses ~8 GB instead of ~2 GB

### When Would You Use Quantization?

**Use quantization when:**

1. **Limited GPU memory**
   ```python
   # Example: Training on a 16GB GPU
   # 7B model in FP32 = 28GB (won't fit!)
   # 7B model in INT4 = 7GB (fits!)
   use_quantization=True, quantization_bits=4
   ```

2. **Inference optimization**
   ```python
   # Deploy model on edge devices
   # Faster inference, lower power consumption
   ```

3. **Training larger models**
   ```python
   # QLoRA: Quantize base model, train LoRA adapters
   # Enables training 70B models on consumer GPUs!
   ```

**Don't use quantization when:**

1. **Mac M-series training** - Not supported, use full precision
2. **Maximum accuracy needed** - Slight accuracy drop from quantization
3. **Sufficient memory** - If model fits comfortably, FP32 is simpler

### Practical Example from Our Project

```python
# On Mac M-series (this project)
model, processor = load_qwen3_vl_with_lora(
    use_quantization=False,  # Disabled for MPS
    device="mps"
)
# Result: ~8GB memory usage, full FP32 precision

# On NVIDIA GPU (hypothetical)
model, processor = load_qwen3_vl_with_lora(
    use_quantization=True,   # Enabled for CUDA
    quantization_bits=4,     # 4-bit quantization
    device="cuda"
)
# Result: ~2GB memory usage, INT4 precision
```

### Key Takeaways

1. **Quantization = Trading precision for memory/speed**
2. **Bitsandbytes = Library that makes quantization easy on CUDA GPUs**
3. **`use_quantization=False` = Your Mac needs full precision (no bitsandbytes support)**
4. **The warning is harmless** - Bitsandbytes gets imported even when unused
5. **On Mac, unified memory helps** - No separate GPU VRAM limit

### Related Concepts to Explore

- **Mixed Precision Training** (FP16/BF16) - Different from quantization
- **LoRA + QLoRA** - Why quantization pairs well with LoRA
- **Gradient Checkpointing** - Another memory-saving technique
- **Model Pruning** - Removing weights vs. quantizing them

---

## 2. Label Masking in Vision-Language Model Training

**Date:** 2025-11-21
**Context:** Training Qwen3-VL with LoRA resulted in catastrophic performance degradation (95% → 15% detection rate)

### What is Label Masking?

Label masking is the process of marking which tokens in a sequence the model should learn to predict during training. In language model training, we typically:
1. Feed the model a sequence of tokens (input + target)
2. Mask the input tokens with `-100` (ignored in loss calculation)
3. Only compute loss on the target (output) tokens

**Why mask?** We don't want the model to learn to predict the prompt - it already knows it. We want it to learn to generate the ANSWER.

```python
# Example sequence
tokens = ["<user>", "What", "is", "2+2", "?", "<assistant>", "4", "</s>"]
labels = [  -100,    -100,   -100,  -100,  -100,      -100,     4,   EOS]
          ^------- masked (no loss) ------^  ^----- train on this ----^
```

### The Critical Bug We Found

In `src/pipeline/training.py` (original lines 196-205), the code attempted to calculate where the assistant's response starts:

```python
# BUGGY CODE (before fix)
prompt_only = self.processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "text", "text": all_inputs[i]["prompt"]}]}],
    tokenize=True,
    add_generation_prompt=True
)
prompt_length = len(prompt_only)
labels[i, :prompt_length] = -100  # Mask first prompt_length tokens
```

**Why this is completely wrong:**

1. **Ignores vision tokens**: Vision-language models convert images into ~100-500 vision tokens that are inserted BEFORE the text prompt
2. **Different tokenization**: Tokenizing just the text prompt produces DIFFERENT tokens than tokenizing the full conversation (with image + assistant response)
3. **Wrong template mode**: Using `add_generation_prompt=True` adds extra tokens that don't exist in the actual training sequence
4. **No correlation**: The calculated `prompt_length` has NO relationship to where the assistant response actually starts

### What Actually Happened During Training

```
Expected masking:
[vision tokens...] [user prompt...] [assistant response...]
^------- mask these -------^ ^--- train on this ---^

Actual (buggy) masking:
[vision tokens...] [user prompt...] [assistant response...]
^- mask first 50 tokens -^ ^--- train on rest ---^
   (calculated from text-only prompt = 50 tokens)
```

**Result:**
- The model trained on PART of the prompt (later vision tokens + beginning of text)
- The model trained on RANDOM portions depending on image size
- The actual assistant response was only partially included in training
- Model learned to output garbage instead of valid JSON bboxes

### The Fix

Search for the actual target tokens in the tokenized sequence:

```python
# FIXED CODE
target_text = bbox_jsons[i]  # e.g., '{"x_min": 0.1, "y_min": 0.2, ...}'
target_tokens = self.processor.tokenizer.encode(target_text, add_special_tokens=False)

# Find where these tokens actually appear
input_ids_list = inputs["input_ids"][i].tolist()
for j in range(len(input_ids_list) - len(target_tokens) + 1):
    if input_ids_list[j:j+len(target_tokens)] == target_tokens:
        assistant_start = j
        break

# Mask everything BEFORE the assistant response
labels[i, :assistant_start] = -100
```

**Why this works:**
1. ✅ Finds the actual position of the target in the tokenized sequence
2. ✅ Accounts for vision tokens automatically (they're already in `input_ids`)
3. ✅ Works regardless of chat template differences
4. ✅ Guarantees we're training on the correct tokens

### How We Discovered This

**Symptoms:**
- Base model: 95% detection rate
- After 2 epochs training: 15% detection rate
- Mean IoU appeared to "improve" (+194%) but only for the 15% that worked
- Trained model outputs were malformed or empty

**Diagnosis Steps:**
1. Compared raw outputs: Trained model produced non-JSON text
2. Inspected tokenization: Calculated mask position didn't align with assistant response
3. Decoded tokens at mask boundary: Found we were masking part of the image, training on part of the prompt
4. Realized vision tokens weren't accounted for

### Key Takeaways

1. **Vision-language models add complexity**: Images become tokens that shift everything
2. **Never calculate positions indirectly**: Always find the actual position in the tokenized sequence
3. **Different template modes produce different tokens**: `add_generation_prompt=True` vs `False` matters
4. **Verify masking**: Always decode and manually check what's masked vs trained
5. **Detection rate > accuracy**: A model that predicts nothing can't have good accuracy
6. **Test on base model first**: If base model works but trained doesn't, suspect training bug

### How to Prevent This

1. **Add validation in training code**:
   ```python
   # Verify mask is correct
   masked_tokens = labels[i][labels[i] != -100]
   assert len(masked_tokens) > 0, "No tokens to train on!"
   assert len(masked_tokens) < len(labels[i]) * 0.5, "Training on too many tokens!"
   ```

2. **Print examples during training**:
   ```python
   if step == 0:  # First step
       print("Example training sample:")
       print("Input:", processor.tokenizer.decode(input_ids[0]))
       print("Labels:", processor.tokenizer.decode(labels[0][labels[0] != -100]))
   ```

3. **Test on tiny dataset first**: 5-10 samples, 1 epoch - should show improvement

### Related Concepts to Explore

- **Attention masking** vs **label masking** (different purposes!)
- **Causal language modeling** - Why we mask past tokens
- **Teacher forcing** - Training technique for sequence models
- **Vision token processing** - How images become token sequences
- **Tokenization consistency** - Why template parameters matter

### Real-World Lesson

This bug wasted:
- ~6 minutes of training time
- Diagnostic time to understand why it failed
- But taught an invaluable lesson about vision-language model internals

**The silver lining:** You now understand tokenization, masking, and vision tokens at a deep level that most practitioners never reach. This knowledge transfers to ANY vision-language model (CLIP, LLaVA, Qwen-VL, GPT-4V concepts, etc.).

---

## 3. Interpreting Detection Metrics & Improving Grounding Models

**Date:** 2025-11-23  
**Context:** Evaluation blocks in `notebooks/train_qwen3_refcoco.ipynb` compare the base Qwen3-VL-2B model against the LoRA-fine-tuned checkpoint on 50 RefCOCO samples.

### How to Read the Metrics

- **Detection Rate** counts how often the model emits a bounding box in the correct JSON format. Base = 90% (45/50), fine-tuned = 100% (50/50). The adapter training eliminated formatting failures so every sample now returns a parseable box.
- **IoU (Intersection-over-Union)** measures overlap quality but only when a box exists. Base mean IoU is 0.1114 (median 0.0), meaning half of the “successful” predictions do not overlap the ground truth at all. Fine-tuning increases mean IoU to 0.3372 and median to 0.1043, so localization is noticeably better but still far from the ≥0.5 IoU that detection systems usually target.
- **Accuracy@τ** reports the share of samples with IoU ≥ τ. Base accuracy@0.5 = 8% (4/50) and @0.75 = 2% (1/50). Fine-tuned accuracy@0.5 jumps to 34% (17/50) and accuracy@0.75 to 28% (14/50), showing that around one-third of predictions now cross the typical “usable box” threshold while high-IoU matches (≥0.75) improved dramatically.

In short, fine-tuning solved structural output errors, raised overlap quality, but most phrases still fail to reach solid IoU, so more work is required before shipping.

### Actionable Improvement Plan

1. **Broaden and balance referring-expression data.**  
   - *Action:* Merge RefCOCO, RefCOCO+, RefCOCOg, and custom captures, ensuring splits cover diverse linguistic patterns, object sizes, and relational cues. Add negative/no-object samples.  
   - *Why:* The median IoU ≈ 0.10 implies the model still “guesses” boxes; richer coverage teaches it to resolve modifiers (“small dog on the left”) rather than defaulting to common regions.
   - *M4 Max:* Feasible; most work is data curation/augmentation that is CPU-bound or relies on lightweight LLM calls. Model training on expanded sets will be slow but manageable if batches stay small.  
   - *RunPod:* Ideal for retraining on the enlarged dataset because higher GPU memory lets you increase batch size and finish epochs faster.

2. **Strengthen supervision with detection-friendly losses.**  
   - *Action:* Decode the model’s bbox tokens back to coordinates during training and apply L1 + GIoU/DIoU losses, or attach an auxiliary regression head fed from the decoder hidden state.  
   - *Why:* Current loss only checks token-level correctness; adding geometric loss directly optimizes spatial accuracy instead of relying on post-hoc IoU evaluation.
   - *M4 Max:* Implementing the loss is fine, but mixed precision/GIoU operations on MPS can be unstable and slow; debugging autograd for custom losses may be tedious.  
   - *RunPod:* Best environment to test new geometric losses since CUDA kernels are mature and you can iterate faster with better profiler tooling.

3. **Tune LoRA capacity and schedules.**  
   - *Action:* Sweep LoRA rank (32→128), α, and dropout; extend training epochs with cosine decay + warmup; monitor eval IoU after each epoch.  
   - *Why:* The modest IoU jump hints at underfitting—more adapter capacity or better scheduling lets the model specialize on grounding without forgetting language skills.
   - *M4 Max:* Practical only for a couple of small experiments due to limited parallelism and slower training; hyperparameter sweeps become multi-day tasks.  
   - *RunPod:* Enables parallel or at least faster sequential sweeps with larger ranks (R=128+) without worrying about unified-memory pressure.

4. **Apply curriculum + augmentation for complex phrases.**  
   - *Action:* Start training on simple noun phrases, then progressively include attribute-rich and relational descriptions; augment by jittering ground-truth boxes, random crops, and LLM paraphrases of the text queries.  
   - *Why:* Many failures surface on long, relational phrases; curriculum learning and augmented phrasing teach invariance to wording and spatial transformations.
   - *M4 Max:* Data-side curriculum/augmentations are inexpensive; the main cost is retraining after each curriculum stage, which is slow but doable.  
   - *RunPod:* Lets you shorten iteration cycles and try more curriculum schedules since each training phase completes sooner.

5. **Improve inference/post-processing.**  
   - *Action:* Clip decoded boxes to image bounds, reject degenerate boxes, sample multiple responses and keep the one whose crop best matches the phrase (CLIP similarity), and log invalid outputs.  
   - *Why:* Even with the current model, simple post-processing can lift IoU by fixing format drift and filtering out obvious mistakes before scoring.
   - *M4 Max:* Entirely feasible locally; operations run on CPU/GPU lightly and help evaluate improvements before pushing to heavier hardware.  
   - *RunPod:* Not necessary unless you need to benchmark large validation sets quickly; otherwise this step is hardware-agnostic.

6. **Close the loop with diagnostics.**  
   - *Action:* Persist per-sample logs: phrase, GT bbox, predicted bbox, IoU, raw response, image hash. Analyze failures by phrase length, object class, and IoU bucket to prioritize fixes.  
   - *Why:* Systematic error analysis turns instinct into evidence, guiding whether to collect more data, tweak losses, or adjust inference.
   - *M4 Max:* Straightforward; exporting CSVs/plots is CPU-bound. Visualization notebooks run fine locally.  
   - *RunPod:* Useful when running large-scale evals (e.g., thousands of samples) where GPU inference plus logging needs to finish fast.

Following this plan increases data diversity, adds geometry-aware training signals, unlocks more capacity in adapters, and tightens inference quality—concrete levers that should push IoU and accuracy@0.5 toward production targets.

---

## 4. Additional Learning Resources

**Date:** 2025-11-23  
**Context:** Curating go-to references for grounding models, LoRA fine-tuning, and detection-style evaluation tactics while iterating on the RefCOCO experiments.

### What is this section?
Hand-picked resources (papers, blogs, repos, lectures) that deepen understanding of referring expression grounding, multi-modal fine-tuning, and metric interpretation.

### Why Does It Matter?
Having a vetted reading list saves time when you need theoretical backing (e.g., for IoU-based losses) or implementation templates (e.g., LoRA sweeps on GPUs) before investing compute cycles.

### Code Examples
- [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO) — End-to-end referring expression detector with strong IoU benchmarks and training scripts you can mine for loss implementations.
- [Qwen-VL Tutorials](https://github.com/QwenLM/Qwen-VL) — Official notebooks showing how to structure prompts, decode bounding boxes, and run LoRA on larger GPUs.

### Key Takeaways
- Keep a mix of *conceptual* (papers, lecture notes) and *practical* (repos, blogs) resources so you can bridge theory ↔ code quickly.
- Revisit these references whenever a metric stalls; chances are there is a known technique (e.g., GIoU loss) already well-documented.

### Related Concepts to Explore
- **Papers:** “Referring Expressions as Programs” (Yu et al. 2018), “Grounded Language-Image Pre-training” (Li et al. 2022).  
- **Talks:** Stanford CS231n guest lecture on grounding & detection metrics.  
- **Blogs:** Hugging Face course chapter on LoRA & QLoRA for vision-language models.  
- **Repos:** OpenGVLab’s [InternVL](https://github.com/OpenGVLab/InternVL) for large-scale VL fine-tuning recipes.  
- **Courses:** fast.ai Part 2 (2023) lessons on multi-modal modeling and prompt engineering pitfalls.

---

## 5. Validation Loss Plateaus & Prompt Flow in RefCOCO

**Date:** 2025-11-23  
**Context:** Training runs logged in `logs/training_history.json` show the validation loss dipping but settling around **0.635** after epoch 3 / step 200, and questions arose about how the referring expression is injected during inference (evaluation cell in `notebooks/train_qwen3_refcoco.ipynb`).

### What does a 0.635 validation loss tell us?

- The loss is the token-level cross-entropy between the generated JSON bbox and the ground truth. A value near 0.63 ≈ perplexity 1.88, meaning the model is still uncertain about the correct coordinate tokens. It is lower than the base model (which hovered near 0.9+) but far from “confident” (≤0.2). In other words, the adapters learned something but have not saturated; 0.63 corresponds to IoUs in the ~0.3 range we see in evaluation.
- The downward trend implies the optimizer is still making progress—no overfitting, yet also no rapid convergence. With small batch sizes on M4 Max, the effective learning signal per step is weak, so the plateau may be caused by limited data, insufficient rank/schedule, or the fact that the loss only considers text similarity (not IoU).

### What should an ML engineer do?

1. **Extend training or add curriculum:** Keep fine-tuning for more epochs (with checkpoint saves) or stage the data so the model sees easier phrases first. Since loss never spikes, we can safely resume from the last checkpoint and push further.
2. **Introduce geometry-aware loss:** Add an auxiliary IoU/L1 loss during training so the adapter directly optimizes spatial accuracy, not just token matching.
3. **Increase effective batch size:** Use gradient accumulation or offload to RunPod to raise the batch size, improving gradient estimates. This often unlocks sharper loss drops, especially for structured generation problems.
4. **Hyperparameter sweep:** The plateau suggests underfitting. Sweeping LoRA rank (R), learning rate, and warmup length may reveal a configuration that continues decreasing loss beyond 0.6.
5. **Data augmentation:** If validation and training losses track closely, more diverse phrases/boxes are needed. Augment text (LLM paraphrases) and boxes (jitter, flips) so the model generalizes better and continues lowering loss.

### Where does the model read the referring expression?

- In `src/pipeline/inference.py:93-147`, each evaluation sample passes `sample['phrase']` into `build_grounding_prompt(phrase)` (imported at the top of the file). The resulting text is injected into the user turn of the chat template together with the image.
- Inside the notebook’s evaluation loop (`notebooks/train_qwen3_refcoco.ipynb:1340-1380` for the base model and `1588-1628` for the trained model), every call to `predict_grounding_bbox(...)` includes that `phrase`. Later, the visualization cell adds `fig.suptitle(f"Phrase: \"{sample['phrase']}\"")`, which is why the figure title matches the instruction the LLM saw.
- Section 22 of the notebook explicitly reconstructs the prompt by calling `build_grounding_prompt(sample_phrase)` and printing the formatted conversation, so you can inspect the exact text the model receives.
- **Important:** The current RefCOCO metadata (`data/refcoco/val/metadata.json`) shows placeholder phrases such as “Please carefully observe the area circled…” instead of true referring expressions. Because the notebook pulls `sample["phrase"]` verbatim, the model is learning to predict boxes from a generic instruction with *no* semantic link to the object. This explains why loss/IoU plateau so early—the text provides no discriminative signal. Fixing the dataset (restore real phrases or synthesize meaningful descriptions) is mandatory before expecting strong performance.

### Key Takeaways

- Validation loss ≈0.63 signals partial learning but not production-grade accuracy; treat it as a cue to run longer, add supervision, or enhance data.
- The referring expression is consistently fed through `build_grounding_prompt` and the chat template, even if the visualization only shows it in the figure title—you can trust that the LLM sees the same text you see because it’s stored in `sample['phrase']` throughout.

### Action Checklist

1. Resume training from the last checkpoint with more epochs or larger effective batches.
2. Prototype an IoU-aware auxiliary loss to push the loss below 0.5.
3. Confirm prompt correctness by re-running the inspection cell (Sec. 22) after any prompt template tweaks.
4. Audit the dataset phrases—if they remain generic placeholders, regenerate/refill them so each sample actually describes the ground-truth region; otherwise the model has no textual grounding signal to learn from.

---

**Format for Future Entries:**

```markdown
## N. [Concept Name]

**Date:** YYYY-MM-DD
**Context:** What you were working on when this came up

### What is [Concept]?
Clear explanation with examples

### Why Does It Matter?
Practical importance

### Code Examples
From this project or general PyTorch

### Key Takeaways
Bullet points of main insights

### Related Concepts to Explore
Links to deeper topics
```
