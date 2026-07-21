"""Load one of the exported complete models.

The models embed a custom MatchedCrop layer and a Lambda wrapping the backbone's
ImageNet preprocess_input function, so both must be supplied at load time.

Usage:
    from load_model import load_final_model
    model = load_final_model("ResNet50", "best_model_resnet50.keras")
    probs = model.predict(images)          # images: float32, (N, 224, 224, 3), 0-255
"""
import keras
import tensorflow as tf

from model_layers import MatchedCrop  # noqa: F401  (registers the custom layer)

PREPROCESS = {
    "ResNet50": tf.keras.applications.resnet50.preprocess_input,
    "ResNet101": tf.keras.applications.resnet.preprocess_input,
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
    "VGG16": tf.keras.applications.vgg16.preprocess_input,
}

CLASS_NAMES = ["C1", "C2", "C3", "C4", "C5"]


def load_final_model(backbone_name, path):
    """Return the trained model for `backbone_name` loaded from `path`."""
    if backbone_name not in PREPROCESS:
        raise ValueError(f"unknown backbone {backbone_name!r}; expected one of {list(PREPROCESS)}")
    return keras.models.load_model(
        path,
        safe_mode=False,
        custom_objects={
            "MatchedCrop": MatchedCrop,
            "preprocess_input": PREPROCESS[backbone_name],
        },
    )
