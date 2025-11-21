"""
Prompt templates for RefCOCO phrase grounding task.
"""


def build_grounding_prompt(phrase: str) -> str:
    """
    Build the instruction prompt for referring expression grounding.

    Given a phrase describing a specific object or region in an image,
    the model should output a JSON bounding box around that region.

    Args:
        phrase: Referring expression (e.g., "the red car on the left")

    Returns:
        Formatted instruction prompt string
    """
    prompt = f"""You are a vision assistant. I will give you an image and a short description that refers to exactly one object or region in the image.

Description: "{phrase}"

Respond with ONLY a JSON object with keys "x_min", "y_min", "x_max", "y_max". All values must be floats between 0 and 1, representing the bounding box for the region matching the description, normalized by image width and height.

JSON:"""

    return prompt


def build_system_message() -> str:
    """
    Returns the system message for the chat template.

    Returns:
        System message string
    """
    return "You are a helpful vision assistant that localizes objects based on text descriptions."


def format_training_messages(phrase: str, image_placeholder: str = "<image>") -> list:
    """
    Format messages for training with Qwen3-VL chat template.

    Args:
        phrase: Referring expression describing the target object
        image_placeholder: Placeholder string for image in message

    Returns:
        List of message dicts for chat template
    """
    return [
        {
            "role": "user",
            "content": f"{image_placeholder}\n{build_grounding_prompt(phrase)}"
        }
    ]


def format_inference_messages(phrase: str, image_placeholder: str = "<image>") -> list:
    """
    Format messages for inference with Qwen3-VL chat template.

    Args:
        phrase: Referring expression describing the target object
        image_placeholder: Placeholder string for image in message

    Returns:
        List of message dicts for chat template
    """
    # Same as training messages, but without the assistant response
    return format_training_messages(phrase, image_placeholder)
