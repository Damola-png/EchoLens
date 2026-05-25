from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_IMAGE_TYPES = ("jpg", "jpeg", "png", "webp")


def load_image_from_bytes(image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    """Load uploaded image bytes as a normalized RGB PIL image and NumPy array."""
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image, np.asarray(image)


def safe_filename(name: str, fallback: str = "photo") -> str:
    """Return a filesystem-safe filename while preserving the original extension."""
    path = Path(name)
    stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in path.stem)
    suffix = path.suffix.lower() if path.suffix else ".jpg"
    return f"{stem or fallback}{suffix}"
