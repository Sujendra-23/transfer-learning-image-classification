"""The prediction service, run against a tiny stub model (no real weights or dataset needed).

The stub is a small functional model with the same shape contract as the real exports: a
224x224x3 float input in the 0-255 range, the repo's custom `MatchedCrop` layer, and five softmax
outputs. It is saved to a real `.keras` file and loaded through `scripts/load_model.py`, so the
custom-layer loading path is exercised. The numbers it produces mean nothing.
"""

import io

import keras
import numpy as np
import pytest
from fastapi.testclient import TestClient
from keras import layers
from PIL import Image

from model_layers import MatchedCrop  # registers the layer
from serving import app as serving_app
from serving.inference import Predictor, decode_image, preprocess

CLASSES = ["C1", "C2", "C3", "C4", "C5"]


def build_stub_model():
    keras.utils.set_random_seed(0)
    inputs = keras.Input((224, 224, 3))
    x = MatchedCrop(200, 200)(inputs)
    x = layers.Rescaling(1 / 255.0)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(5, activation="softmax")(x)
    return keras.Model(inputs, outputs)


@pytest.fixture(scope="module")
def stub_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("stub") / "stub.keras"
    build_stub_model().save(path)
    return str(path)


@pytest.fixture(scope="module")
def client(stub_path):
    """The real app, loading the stub from MODEL_PATH-style config at startup."""
    app = serving_app.create_app(model_path=stub_path, backbone="ResNet50")
    with TestClient(app) as test_client:
        yield test_client


def image_bytes(size=(64, 48), mode="RGB", fmt="PNG", color=(200, 30, 30)):
    buffer = io.BytesIO()
    Image.new(mode, size, color if mode == "RGB" else 128).save(buffer, format=fmt)
    return buffer.getvalue()


def post(client, data, name="x.png", **kwargs):
    return client.post("/predict", files={"file": (name, data, "application/octet-stream")}, **kwargs)


def test_health_reports_the_loaded_model(client):
    body = client.get("/health").json()
    assert body == {
        "status": "ok", "backbone": "ResNet50", "classes": CLASSES, "input_size": [224, 224],
    }


def test_predict_returns_a_probability_distribution_over_five_classes(client):
    response = post(client, image_bytes())
    assert response.status_code == 200
    body = response.json()
    assert list(body["probabilities"]) == CLASSES
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert body["predicted_class"] == max(body["probabilities"], key=body["probabilities"].get)
    assert body["confidence"] == pytest.approx(body["probabilities"][body["predicted_class"]])
    assert body["backbone"] == "ResNet50" and body["latency_ms"] > 0


def test_service_matches_calling_the_model_directly_on_the_same_preprocessing(client, stub_path):
    data = image_bytes((300, 200), color=(10, 200, 90))
    served = post(client, data).json()["probabilities"]
    model = keras.models.load_model(stub_path, safe_mode=False, custom_objects={"MatchedCrop": MatchedCrop})
    direct = model.predict_on_batch(preprocess(data)[None, ...])[0]
    assert [served[c] for c in CLASSES] == pytest.approx(list(direct), abs=1e-5)


@pytest.mark.parametrize("mode,fmt", [("RGB", "JPEG"), ("RGB", "PNG"), ("L", "PNG"), ("RGBA", "PNG"), ("P", "PNG")])
def test_common_image_modes_and_formats_are_accepted(client, mode, fmt):
    buffer = io.BytesIO()
    Image.new(mode, (40, 40)).save(buffer, format=fmt)
    assert post(client, buffer.getvalue()).status_code == 200


def test_preprocess_gives_a_224_float32_array_in_0_255():
    array = preprocess(image_bytes((500, 100), color=(255, 255, 255)))
    assert array.shape == (224, 224, 3) and array.dtype == np.float32
    assert array.max() == pytest.approx(255, abs=1e-3)


def test_grayscale_and_rgba_decode_to_three_channels():
    for mode in ("L", "RGBA", "P"):
        buffer = io.BytesIO()
        Image.new(mode, (8, 8)).save(buffer, format="PNG")
        assert decode_image(buffer.getvalue()).shape == (8, 8, 3)


def test_non_image_upload_is_415(client):
    response = post(client, b"this is not an image", name="notes.txt")
    assert response.status_code == 415
    assert "not a readable image" in response.json()["detail"]


def test_truncated_image_is_415(client):
    assert post(client, image_bytes()[:30]).status_code == 415


def test_empty_upload_is_400(client):
    assert post(client, b"").status_code == 400


def test_missing_file_field_is_422(client):
    assert client.post("/predict").status_code == 422


def test_oversized_upload_is_413(stub_path):
    app = serving_app.create_app(model_path=stub_path, backbone="ResNet50", max_upload_bytes=1000)
    with TestClient(app) as small:
        assert post(small, b"x" * 1001).status_code == 413
        assert post(small, image_bytes((4, 4))).status_code == 200  # still works under the limit


def test_health_and_predict_are_503_until_a_model_is_loaded():
    app = serving_app.create_app(predictor=None, model_path=None)
    app.state.predictor = None
    # Bypass lifespan (which would refuse to start without a model) to see the unloaded state.
    unloaded = TestClient(app)
    assert unloaded.get("/health").status_code == 503
    assert post(unloaded, image_bytes()).status_code == 503


def test_startup_fails_fast_without_or_with_a_wrong_model_path(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_PATH"):
        with TestClient(serving_app.create_app()):
            pass
    with pytest.raises(RuntimeError, match="does not exist"):
        with TestClient(serving_app.create_app(model_path="/nope/model.keras")):
            pass


def test_unknown_backbone_is_rejected_at_startup(stub_path):
    with pytest.raises(RuntimeError, match="BACKBONE"):
        with TestClient(serving_app.create_app(model_path=stub_path, backbone="ResNet9000")):
            pass


def test_predictor_wraps_any_model_with_predict_on_batch():
    class Fake:
        input_shape = (None, 224, 224, 3)

        def predict_on_batch(self, batch):
            return np.tile([0.1, 0.2, 0.3, 0.3, 0.1], (len(batch), 1))

    result = Predictor(Fake(), CLASSES, "Fake").classify(image_bytes())
    assert result["predicted_class"] == "C3"  # argmax picks the first of the tied maxima
    assert result["confidence"] == pytest.approx(0.3)
