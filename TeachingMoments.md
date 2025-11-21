# Teaching Moments: PyTorch Mastery

This document serves as a living knowledge base - a collection of concepts, techniques, and insights encountered while working on this project. Each entry captures a learning moment, providing deep explanations that build toward PyTorch mastery.

---

## Table of Contents

1. [Model Quantization & Bitsandbytes](#1-model-quantization--bitsandbytes)
2. [Label Masking in Vision-Language Model Training](#2-label-masking-in-vision-language-model-training)

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
