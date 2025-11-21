"""
Qwen3-VL model loading and LoRA/QLoRA configuration.

This module handles:
- Loading Qwen3-VL-2B-Instruct model and processor
- Setting up LoRA/QLoRA for efficient fine-tuning
- Preparing the model for training on Mac M-series (MPS)
"""
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Tuple, Optional, Dict, Any


def load_qwen3_vl_with_lora(
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    use_quantization: bool = True,
    quantization_bits: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    device: str | torch.device = "mps",
    max_memory: Optional[Dict] = None
) -> Tuple[Any, Any]:
    """
    Load Qwen3-VL-2B-Instruct model with LoRA/QLoRA configuration.

    This function:
    1. Loads the base Qwen3-VL model (optionally with quantization)
    2. Wraps it with LoRA adapters for efficient fine-tuning
    3. Freezes base model parameters and only trains LoRA adapters

    Args:
        model_name: HuggingFace model identifier
        use_quantization: Whether to use 4-bit/8-bit quantization (QLoRA)
        quantization_bits: 4 or 8 bit quantization
        lora_r: LoRA rank (dimension of low-rank matrices)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout rate for LoRA layers
        target_modules: List of module names to apply LoRA to.
                       If None, uses default for Qwen models
        device: Device to load model on ('mps', 'cpu', 'cuda', or torch.device object)
        max_memory: Optional dict specifying max memory per device

    Returns:
        Tuple of (model, processor):
        - model: PEFT-wrapped model ready for training
        - processor: Multimodal processor for images + text

    Notes on Qwen3-VL architecture:
    - Vision encoder processes images → vision features
    - Vision features are projected and fed into the language model
    - Language model generates text conditioned on vision + text
    - We apply LoRA primarily to language model attention layers
    """
    # Normalize device to string for comparisons
    device_str = device if isinstance(device, str) else device.type

    print(f"Loading Qwen3-VL model: {model_name}")
    print(f"Device: {device_str}")
    print(f"Quantization: {use_quantization} ({quantization_bits}-bit)")
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    # ====================
    # 1. Load Processor
    # ====================
    # The processor handles both image preprocessing and text tokenization
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # ====================
    # 2. Setup Quantization Config (if using QLoRA)
    # ====================
    quantization_config = None
    if use_quantization:
        # Note: BitsAndBytes may not work well with MPS
        # For Mac M-series, we fall back to full precision
        if device_str == "mps":
            print("Warning: Quantization not well supported on MPS. Using full precision.")
            print("For memory constraints on MPS, use model='Qwen/Qwen3-VL-2B-Instruct-FP8'")
            use_quantization = False
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=(quantization_bits == 4),
                load_in_8bit=(quantization_bits == 8),
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

    # ====================
    # 3. Load Base Model
    # ====================
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device_str != "mps" else torch.float32,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if device_str == "mps":
        # For MPS, we load on CPU first then move to MPS
        # This avoids some compatibility issues
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = "auto"

    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Move to device if using MPS
    if device_str == "mps":
        model = model.to(device_str)

    # ====================
    # 4. Prepare for k-bit training (if quantized)
    # ====================
    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    # ====================
    # 5. Setup LoRA Configuration
    # ====================
    # Qwen3-VL uses Qwen2 architecture for the language model
    # Target modules typically include:
    # - q_proj, k_proj, v_proj (attention query, key, value)
    # - o_proj (attention output projection)
    # - gate_proj, up_proj, down_proj (MLP layers)

    if target_modules is None:
        # Default: target attention layers
        # For more aggressive tuning, add MLP layers too
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # Uncomment to also train MLP layers (more parameters):
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ====================
    # 6. Wrap Model with LoRA
    # ====================
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    print(f"\nModel loaded successfully on {device_str}")
    print(f"Model type: {type(model)}")

    return model, processor


def count_parameters(model) -> Dict[str, int]:
    """
    Count total and trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "trainable_pct": 100 * trainable_params / total_params if total_params > 0 else 0
    }


def save_lora_weights(model, output_path: str):
    """
    Save only the LoRA adapter weights.

    Args:
        model: PEFT-wrapped model
        output_path: Path to save LoRA weights
    """
    model.save_pretrained(output_path)
    print(f"LoRA weights saved to {output_path}")


def load_lora_weights(model, lora_weights_path: str):
    """
    Load LoRA adapter weights into a PEFT model.

    Args:
        model: PEFT-wrapped model
        lora_weights_path: Path to LoRA weights

    Returns:
        Model with loaded LoRA weights
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, lora_weights_path)
    print(f"LoRA weights loaded from {lora_weights_path}")
    return model


def merge_and_save_full_model(model, processor, output_path: str):
    """
    Merge LoRA weights into base model and save the full model.

    Args:
        model: PEFT-wrapped model
        processor: Processor to save alongside model
        output_path: Path to save merged model
    """
    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()

    # Save merged model and processor
    merged_model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")
