"""
Training loop for fine-tuning Qwen3-VL on main subject bounding box detection.
"""
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Optional, Any
import time

from .prompts import build_main_subject_prompt


class Trainer:
    """
    Trainer for fine-tuning Qwen3-VL with LoRA on bounding box detection.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: str = "mps",
        output_dir: Path = Path("outputs/checkpoints"),
        log_dir: Path = Path("outputs/logs"),
        logging_steps: int = 10,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PEFT-wrapped Qwen3-VL model
            processor: Qwen3-VL processor
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for lr scheduler
            gradient_accumulation_steps: Accumulate gradients over N steps
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
            output_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps (if None, eval at end of epoch)
            save_steps: Save checkpoint every N steps (if None, save at end of epoch)
        """
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps

        # Setup directories
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Setup lr scheduler
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.training_history = []

        print(f"\nTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Steps per epoch: {len(train_dataloader)}")
        print(f"  Total training steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")

    def prepare_batch(self, batch: Dict) -> Dict:
        """
        Prepare a batch for training with Qwen3-VL.

        This function:
        1. Formats images and text into Qwen3-VL's expected format
        2. Tokenizes inputs and targets
        3. Creates labels with -100 for prompt tokens (ignored in loss)

        Args:
            batch: Dictionary with 'image', 'gt_box_json' keys

        Returns:
            Dictionary with processed inputs ready for model
        """
        images = batch["image"]
        gt_box_jsons = batch["gt_box_json"]
        batch_size = len(images)

        # Build prompt (same for all examples)
        prompt = build_main_subject_prompt()

        # Prepare messages for each example
        # Qwen3-VL expects a conversation format
        all_inputs = []

        for i in range(batch_size):
            # Format as conversation
            # User message contains image + prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[i]},
                        {"type": "text", "text": prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": gt_box_jsons[i]}
                    ]
                }
            ]

            # Use processor to format
            # This creates input_ids with the full conversation
            # We'll need to mask out the prompt part in labels
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            all_inputs.append({"text": text, "images": images[i]})

        # Process batch
        # The processor handles both image preprocessing and text tokenization
        inputs = self.processor(
            text=[inp["text"] for inp in all_inputs],
            images=[inp["images"] for inp in all_inputs],
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Create labels
        # We need to mask the prompt tokens so only the assistant response is trained
        labels = inputs["input_ids"].clone()

        # For each example, find where the assistant response starts
        # and mask everything before it with -100
        for i in range(batch_size):
            # Find the assistant response start
            # The processor should have a special token or we need to identify it
            # For simplicity, we'll mask based on the prompt length
            # This is a simplified approach - you may need to adjust based on
            # Qwen3-VL's exact tokenization behavior

            # Tokenize just the prompt to find its length
            prompt_only = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                tokenize=True,
                add_generation_prompt=True
            )
            prompt_length = len(prompt_only)

            # Mask prompt tokens
            labels[i, :prompt_length] = -100

        inputs["labels"] = labels

        return inputs

    def training_step(self, batch: Dict) -> float:
        """
        Perform a single training step.

        Args:
            batch: Batch of data

        Returns:
            Loss value
        """
        # Prepare inputs
        inputs = self.prepare_batch(batch)

        # Forward pass
        outputs = self.model(**inputs)
        loss = outputs.loss

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        return loss.item() * self.gradient_accumulation_steps

    def train_epoch(self, epoch: int):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs}"
        )

        for step, batch in enumerate(progress_bar):
            # Training step
            loss = self.training_step(batch)
            epoch_loss += loss
            num_batches += 1

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}"
                    })

                    # Log to history
                    self.training_history.append({
                        "epoch": epoch,
                        "step": self.global_step,
                        "loss": avg_loss,
                        "learning_rate": lr
                    })

                # Evaluation
                if self.eval_steps and self.global_step % self.eval_steps == 0:
                    if self.val_dataloader:
                        self.evaluate()
                        self.model.train()

                # Checkpointing
                if self.save_steps and self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

        return epoch_loss / num_batches

    def evaluate(self):
        """
        Run evaluation on validation set.

        Note: This just computes validation loss.
        For IoU metrics, use the eval.py module.
        """
        if not self.val_dataloader:
            return

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                inputs = self.prepare_batch(batch)
                outputs = self.model(**inputs)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"\nValidation Loss: {avg_loss:.4f}")

        # Log validation metrics
        self.training_history.append({
            "epoch": self.current_epoch,
            "step": self.global_step,
            "val_loss": avg_loss
        })

        return avg_loss

    def train(self):
        """
        Run the full training loop.
        """
        print("\n" + "=" * 50)
        print("Starting training...")
        print("=" * 50)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            avg_loss = self.train_epoch(epoch)

            print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

            # Evaluate at end of epoch
            if self.val_dataloader:
                self.evaluate()

            # Save checkpoint at end of epoch
            self.save_checkpoint(f"epoch_{epoch + 1}")

        # Training complete
        end_time = time.time()
        training_time = end_time - start_time

        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Total time: {training_time / 60:.2f} minutes")
        print("=" * 50)

        # Save final model
        self.save_checkpoint("final")

        # Save training history
        self.save_training_history()

    def save_checkpoint(self, name: str):
        """
        Save model checkpoint.

        Args:
            name: Checkpoint name
        """
        checkpoint_path = self.output_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        self.model.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        torch.save(state, checkpoint_path / "training_state.pt")

        print(f"Checkpoint saved to {checkpoint_path}")

    def save_training_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / "training_history.json"

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"Training history saved to {history_path}")
