"""
Evaluation module for RefCOCO phrase grounding with IoU metrics.
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..data.box_utils import bbox_iou, json_to_bbox, extract_json_from_text
from .prompts import build_grounding_prompt


class Evaluator:
    """
    Evaluator for bounding box detection with IoU metrics.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: str = "mps",
        iou_thresholds: List[float] = [0.5, 0.75],
        max_new_tokens: int = 100,
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model (PEFT or merged)
            processor: Qwen3-VL processor
            device: Device to run evaluation on
            iou_thresholds: IoU thresholds for success rate computation
            max_new_tokens: Maximum tokens to generate
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.iou_thresholds = sorted(iou_thresholds)
        self.max_new_tokens = max_new_tokens

        self.model.eval()

    def predict_bbox(
        self,
        image,
        phrase: str,
        return_raw_text: bool = True
    ) -> Tuple[Optional[Tuple[float, float, float, float]], str]:
        """
        Predict bounding box for a single image given a referring expression phrase.

        Args:
            image: PIL Image
            phrase: Referring expression (e.g., "the red car on the left")
            return_raw_text: Whether to return raw generated text

        Returns:
            Tuple of (bbox_norm, raw_text):
            - bbox_norm: Normalized bbox (x_min, y_min, x_max, y_max) or None if parsing failed
            - raw_text: Raw generated text
        """
        # Build phrase-conditional prompt
        prompt = build_grounding_prompt(phrase)

        # Format messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode output
        # Only decode the generated part (exclude input)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        # Try to extract JSON from the raw text
        json_str = extract_json_from_text(raw_text)

        # Parse JSON to bbox
        if json_str:
            bbox_norm = json_to_bbox(json_str)
        else:
            # If no JSON found, try parsing the raw text directly
            bbox_norm = json_to_bbox(raw_text)

        if return_raw_text:
            return bbox_norm, raw_text
        else:
            return bbox_norm

    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        save_predictions: bool = False,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            max_samples: Maximum number of samples to evaluate (None = all)
            save_predictions: Whether to save predictions to file
            output_path: Path to save predictions (required if save_predictions=True)

        Returns:
            Dictionary with evaluation metrics
        """
        print("\nRunning evaluation...")

        all_ious = []
        all_predictions = []
        num_parsed = 0
        num_failed = 0

        # Initialize counters for each threshold
        success_counts = {thresh: 0 for thresh in self.iou_thresholds}

        num_samples = 0
        max_samples = max_samples or float('inf')

        for batch in tqdm(dataloader, desc="Evaluating"):
            if num_samples >= max_samples:
                break

            images = batch["image"]
            phrases = batch["phrase"]
            gt_boxes = batch["bbox_norm"]
            widths = batch["width"]
            heights = batch["height"]

            # Process each image in batch
            for i in range(len(images)):
                if num_samples >= max_samples:
                    break

                image = images[i]
                phrase = phrases[i]
                gt_box = gt_boxes[i]

                # Predict with phrase
                pred_box, raw_text = self.predict_bbox(image, phrase, return_raw_text=True)

                # Check if parsing succeeded
                if pred_box is None:
                    num_failed += 1
                    iou = 0.0
                else:
                    num_parsed += 1
                    # Compute IoU
                    iou = bbox_iou(gt_box, pred_box)
                    all_ious.append(iou)

                    # Check thresholds
                    for thresh in self.iou_thresholds:
                        if iou >= thresh:
                            success_counts[thresh] += 1

                # Save prediction info
                if save_predictions:
                    all_predictions.append({
                        "sample_idx": num_samples,
                        "gt_box": gt_box,
                        "pred_box": pred_box,
                        "iou": iou,
                        "raw_text": raw_text,
                        "width": widths[i].item() if torch.is_tensor(widths[i]) else widths[i],
                        "height": heights[i].item() if torch.is_tensor(heights[i]) else heights[i],
                    })

                num_samples += 1

        # Compute metrics
        metrics = self._compute_metrics(
            all_ious,
            success_counts,
            num_samples,
            num_parsed,
            num_failed
        )

        # Save predictions if requested
        if save_predictions and output_path:
            self._save_predictions(all_predictions, metrics, output_path)

        # Print results
        self._print_metrics(metrics)

        return metrics

    def _compute_metrics(
        self,
        all_ious: List[float],
        success_counts: Dict[float, int],
        num_samples: int,
        num_parsed: int,
        num_failed: int
    ) -> Dict:
        """Compute evaluation metrics."""
        metrics = {
            "num_samples": num_samples,
            "num_parsed": num_parsed,
            "num_failed": num_failed,
            "parse_rate": num_parsed / num_samples if num_samples > 0 else 0.0,
        }

        # IoU metrics (only for successfully parsed predictions)
        if all_ious:
            metrics["mean_iou"] = np.mean(all_ious)
            metrics["median_iou"] = np.median(all_ious)
            metrics["std_iou"] = np.std(all_ious)
            metrics["min_iou"] = np.min(all_ious)
            metrics["max_iou"] = np.max(all_ious)
        else:
            metrics["mean_iou"] = 0.0
            metrics["median_iou"] = 0.0
            metrics["std_iou"] = 0.0
            metrics["min_iou"] = 0.0
            metrics["max_iou"] = 0.0

        # Success rates at different thresholds
        for thresh in self.iou_thresholds:
            success_rate = success_counts[thresh] / num_samples if num_samples > 0 else 0.0
            metrics[f"success_at_{thresh}"] = success_rate

        return metrics

    def _print_metrics(self, metrics: Dict):
        """Print evaluation metrics in a readable format."""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Total samples: {metrics['num_samples']}")
        print(f"Successfully parsed: {metrics['num_parsed']} ({metrics['parse_rate']:.1%})")
        print(f"Failed to parse: {metrics['num_failed']}")
        print("\nIoU Statistics (on successfully parsed predictions):")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"  Median IoU: {metrics['median_iou']:.4f}")
        print(f"  Std IoU: {metrics['std_iou']:.4f}")
        print(f"  Min IoU: {metrics['min_iou']:.4f}")
        print(f"  Max IoU: {metrics['max_iou']:.4f}")
        print("\nSuccess Rates:")
        for thresh in self.iou_thresholds:
            rate = metrics[f"success_at_{thresh}"]
            print(f"  IoU ≥ {thresh}: {rate:.1%}")
        print("=" * 50)

    def _save_predictions(
        self,
        predictions: List[Dict],
        metrics: Dict,
        output_path: Path
    ):
        """Save predictions and metrics to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metrics": metrics,
            "predictions": predictions
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\nPredictions saved to {output_path}")


def evaluate_from_checkpoint(
    checkpoint_path: Path,
    dataloader: DataLoader,
    device: str = "mps",
    **kwargs
) -> Dict:
    """
    Convenience function to load a checkpoint and evaluate.

    Args:
        checkpoint_path: Path to model checkpoint
        dataloader: DataLoader for evaluation
        device: Device to use
        **kwargs: Additional arguments for Evaluator

    Returns:
        Evaluation metrics dictionary
    """
    from .model_qwen3 import load_qwen3_vl_with_lora, load_lora_weights

    # Load base model
    model, processor = load_qwen3_vl_with_lora(device=device)

    # Load LoRA weights
    model = load_lora_weights(model, str(checkpoint_path))

    # Create evaluator
    evaluator = Evaluator(model, processor, device=device, **kwargs)

    # Run evaluation
    metrics = evaluator.evaluate_dataset(dataloader)

    return metrics
