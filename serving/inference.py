"""Image decoding, preprocessing and prediction, kept separate from the HTTP layer.

Preprocessing mirrors the notebook's input pipeline (`decode_and_resize`): decode to 3 channels,
resize to 224x224 with `tf.image.resize(..., antialias=True)`, float32 in the 0-255 range. The
exported models contain the backbone's ImageNet `preprocess_input` and the centre crop inside the
graph, so nothing else is applied here.
"""

from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image, UnidentifiedImageError

# Reject decompression bombs before decoding; far above any real photo, far below a bomb.
Image.MAX_IMAGE_PIXELS = 50_000_000

INPUT_SIZE = (224, 224)


class ImageError(ValueError):
    """The upload is not an image the service can read. The message is safe to return."""


def decode_image(data: bytes) -> np.ndarray:
    """Return an (H, W, 3) uint8 array. EXIF orientation is ignored, as in training."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageError("File is not a readable image") from exc


def resize(array: np.ndarray, size: tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    """Bilinear resize with antialiasing, identical to the notebook's `tf.image.resize` call."""
    import tensorflow as tf

    resized = tf.image.resize(array, size, antialias=True)
    return tf.cast(resized, tf.float32).numpy()


def preprocess(data: bytes, size: tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    return resize(decode_image(data), size)


class Predictor:
    """Thin wrapper around a Keras model: batches in, class probabilities out.

    Calls are serialized with a lock. A single process serves one model on one device, so
    concurrent calls would only contend; scale by adding processes or replicas instead.
    """

    def __init__(self, model, class_names: list[str], backbone: str) -> None:
        self.model = model
        self.class_names = class_names
        self.backbone = backbone
        self.input_size = tuple(model.input_shape[1:3]) if hasattr(model, "input_shape") else INPUT_SIZE
        self._lock = threading.Lock()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        with self._lock:
            return np.asarray(self.model.predict_on_batch(batch), dtype=np.float32)

    def classify(self, data: bytes) -> dict:
        probabilities = self.predict(preprocess(data, self.input_size)[None, ...])[0]
        best = int(np.argmax(probabilities))
        return {
            "predicted_class": self.class_names[best],
            "confidence": float(probabilities[best]),
            "probabilities": {
                name: float(p) for name, p in zip(self.class_names, probabilities, strict=True)
            },
            "backbone": self.backbone,
        }
