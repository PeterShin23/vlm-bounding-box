"""
Prompt templates for the main subject bounding box task.
"""


def build_main_subject_prompt() -> str:
    """
    Returns the fixed instruction string for the main subject detection task.

    The model should respond with ONLY a JSON object containing normalized
    bounding box coordinates.

    Returns:
        Instruction prompt string
    """
    prompt = """You are a vision assistant. You are given an image. Your job is to find the single most important, visually salient subject in the image and draw a tight bounding box around it.

Respond with ONLY a JSON object containing four keys: "x_min", "y_min", "x_max", "y_max". All values must be floats between 0 and 1, representing normalized coordinates relative to the image width and height.

JSON:"""

    return prompt


def build_system_message() -> str:
    """
    Returns the system message for the chat template.

    Returns:
        System message string
    """
    return "You are a helpful vision assistant that detects main subjects in images."


def format_training_messages(image_placeholder: str = "<image>") -> list:
    """
    Format messages for training with Qwen3-VL chat template.

    Args:
        image_placeholder: Placeholder string for image in message

    Returns:
        List of message dicts for chat template
    """
    return [
        {
            "role": "user",
            "content": f"{image_placeholder}\n{build_main_subject_prompt()}"
        }
    ]


def format_inference_messages(image_placeholder: str = "<image>") -> list:
    """
    Format messages for inference with Qwen3-VL chat template.

    Args:
        image_placeholder: Placeholder string for image in message

    Returns:
        List of message dicts for chat template
    """
    # Same as training messages, but without the assistant response
    return format_training_messages(image_placeholder)
