from __future__ import annotations

import base64
import os
import uuid

from app.core.config import Settings

__all__ = ["save_image_base64", "save_image_bytes"]


def _ensure_outputs_dir(base_dir: str) -> str:
    output_directory = os.path.join(base_dir, "outputs")
    os.makedirs(output_directory, exist_ok=True)
    return output_directory


def save_image_base64(
    image_data: str, settings: Settings, file_extension: str | None
) -> str | None:
    """Decode base64 image data and persist it to disk.

    Returns absolute file path if saved successfully; otherwise None.
    """
    try:
        decoded = base64.b64decode(image_data, validate=True)
        outputs = _ensure_outputs_dir(settings.tmp_dir)
        ext = (file_extension or "png").lower()
        filename = f"diagram_{uuid.uuid4()}.{ext}"
        saved_path = os.path.join(outputs, filename)
        with open(saved_path, "wb") as f:
            f.write(decoded)
        return saved_path
    except Exception:
        return None


def save_image_bytes(data: bytes, base_tmp_dir: str) -> str:
    """Persist raw image bytes to disk under base_tmp_dir/outputs and return path."""
    outputs = _ensure_outputs_dir(base_tmp_dir)
    filename = f"diagram_{uuid.uuid4()}.png"
    path = os.path.join(outputs, filename)
    with open(path, "wb") as f:
        f.write(data)
    return path
