"""Generate visual notes/captions using a VL model from LM Studio."""

import base64
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are a visual analyst. Describe the key visual elements in this image "
    "concisely (1-2 sentences). Focus on objects, people, actions, colors, and "
    "scene composition. Do NOT mention any text in the image."
)


def encode_image_b64(image_path: str) -> str:
    """Read an image file and return a base64 data URL.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 data URL string (e.g. 'data:image/png;base64,...').
    """
    ext = Path(image_path).suffix.lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def generate_caption(
    client,
    image_path: str,
    model: str = "qwen/qwen3-vl-30b",
    system_prompt: str | None = None,
) -> str:
    """Send image to LM Studio VL model and return caption.

    Args:
        client: OpenAI-compatible client configured for LM Studio.
        image_path: Path to the image file.
        model: VL model name (must support vision).
        system_prompt: Optional custom system prompt.

    Returns:
        Caption string from the model's response.
    """
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    image_url = encode_image_b64(image_path)

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the visual content of this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def generate_captions(
    client,
    image_paths: list[str],
    model: str = "qwen/qwen3-vl-30b",
    batch_size: int = 4,
) -> dict[str, str]:
    """Generate captions for multiple images.

    Args:
        client: OpenAI-compatible client configured for LM Studio.
        image_paths: List of image file paths.
        model: VL model name.
        batch_size: Not used for chat API (kept for future batching).

    Returns:
        Mapping of image path to caption string.
    """
    from tqdm import tqdm

    captions: dict[str, str] = {}
    for path in tqdm(image_paths, desc="Generating captions"):
        try:
            captions[path] = generate_caption(client, path, model=model)
        except Exception as e:
            # Skip failed captions — error strings would produce garbage vectors.
            pass

    return captions
