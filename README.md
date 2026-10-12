# Transfer Learning for Image Classification

Classifies images across five classes (`C1`–`C5`) using four ImageNet-pretrained
convolutional backbones — **ResNet50, ResNet101, EfficientNetB0, VGG16** — as frozen
feature extractors, with a small regularized classification head trained on top.

Built with **TensorFlow 2.16 / Keras 3**. GPU training on Apple silicon uses the
`tensorflow-metal` plug-in.

## Contents

- `notebook/Bharadwaj_Sujendra_Giri_Prasad_Final_Project.ipynb` — full analysis: data
  exploration, class-wise split, augmentation, the four transfer-learning models, training
  with early stopping, and evaluation.
- `scripts/model_layers.py` — custom `MatchedCrop` layer (random crop while training,
  centre crop while evaluating).
- `scripts/load_model.py` — helper to load the exported `.keras` models.

## Approach

- Backbones are frozen; only a 256-unit ReLU layer, batch normalization, 20% dropout, and a
  softmax output are trained. VGG16 is cut at its `fc2` penultimate layer; the others use
  global average pooling.
- Six augmentations (crop, zoom, rotate, flip, contrast, translate) are applied inside the
  model, active only during training.
- Adam optimizer, categorical cross-entropy, L2 on the hidden layer, up to 100 epochs with
  early stopping on validation loss (best-weight restoration).

## Results

All metrics are macro-averaged across the five classes.

### Test set (ranked by macro F1)

| Model | F1 | Accuracy | AUC | Precision | Recall |
|-------|---:|---:|---:|---:|---:|
| ResNet50 | 0.8119 | 0.7971 | 0.9484 | 0.8569 | 0.7958 |
| ResNet101 | 0.8018 | 0.7867 | 0.9474 | 0.8400 | 0.7909 |
| EfficientNetB0 | 0.7763 | 0.7637 | 0.9342 | 0.8284 | 0.7438 |
| VGG16 | 0.6505 | 0.6875 | 0.8917 | 0.7584 | 0.6208 |

### Training

| Model | F1 | Accuracy | AUC | Precision | Recall |
|-------|---:|---:|---:|---:|---:|
| ResNet50 | 0.8372 | 0.8183 | 0.9594 | 0.8806 | 0.8198 |
| ResNet101 | 0.8318 | 0.8128 | 0.9585 | 0.8690 | 0.8135 |
| EfficientNetB0 | 0.8013 | 0.7836 | 0.9469 | 0.8539 | 0.7679 |
| VGG16 | 0.6628 | 0.6978 | 0.8959 | 0.7734 | 0.6283 |

### Validation

| Model | F1 | Accuracy | AUC | Precision | Recall |
|-------|---:|---:|---:|---:|---:|
| ResNet50 | 0.7961 | 0.7873 | 0.9434 | 0.8398 | 0.7787 |
| ResNet101 | 0.7943 | 0.7911 | 0.9454 | 0.8384 | 0.7717 |
| EfficientNetB0 | 0.7694 | 0.7626 | 0.9331 | 0.8370 | 0.7314 |
| VGG16 | 0.6410 | 0.6919 | 0.8902 | 0.7405 | 0.6118 |

ResNet50 and ResNet101 finish effectively tied at the top (within 0.01 macro F1); VGG16 is
clearly the weakest. Every model handles classes C1 and C5 well and struggles most with C2.

## Setup

```bash
pip install -r requirements.txt
```

Place the image dataset under `data/` as `data/C1/`, `data/C2/`, … before running the
notebook. Data files are not committed here due to size.

## Author

Sujendra Giri Prasad Bharadwaj

## Serving a trained model

`serving/` is a FastAPI service for one exported `.keras` model; `Dockerfile` packages it. The
models and dataset are not in this repo (see Setup), so the image does not contain a model: you
mount one and set `MODEL_PATH`.

```bash
pip install tensorflow==2.16.2 -r serving/requirements.txt
MODEL_PATH=/path/to/best_model_resnet50.keras BACKBONE=ResNet50 \
  uvicorn serving.app:app --port 8000

curl localhost:8000/health
curl -F file=@some_image.jpg localhost:8000/predict
```

| Endpoint | |
| --- | --- |
| `GET /health` | `200` with the backbone, class names and input size once the model is loaded; `503` before. The process exits at startup if `MODEL_PATH` is missing or wrong, rather than serving errors. |
| `POST /predict` | Multipart upload, field `file`. Returns `predicted_class`, `confidence`, `probabilities` for `C1`-`C5`, `backbone`, `latency_ms`. `400` empty upload, `413` over `MAX_UPLOAD_BYTES` (default 8 MB), `415` not a readable image, `422` no file. |

The model is loaded through `scripts/load_model.py`, so the custom `MatchedCrop` layer and the
backbone's `preprocess_input` are registered the same way as in the notebook. Uploads are decoded to
3 channels and resized to 224x224 with `tf.image.resize(..., antialias=True)`, matching the
notebook's `decode_and_resize`; the exported graph then applies the ImageNet preprocessing and the
centre crop itself. Run one uvicorn process per container: predictions are serialized behind a lock,
so extra workers only multiply memory.

```bash
docker build -t tl-classifier .
docker run --rm -p 8000:8000 -v /path/to/models:/models:ro \
  -e MODEL_PATH=/models/best_model_resnet50.keras -e BACKBONE=ResNet50 tl-classifier
```

### Latency benchmark

```bash
python scripts/benchmark_latency.py --model best_model_resnet50.keras --backbone ResNet50
python scripts/benchmark_latency.py --model ... --backbone ResNet50 --device cpu     # hide the GPU
python scripts/benchmark_latency.py --http http://localhost:8000/predict --image img.jpg
```

Model mode times `predict_on_batch` at batch sizes 1, 8 and 32 with 5 warm-up iterations per size
excluded and 30 timed ones, and reports p50 and p95 batch latency, per-image latency and
throughput. Inputs are random pixels unless `--images DIR` is given. HTTP mode times the full
`/predict` round trip one image at a time. The script prints the hardware and TensorFlow device it ran
on; always quote that with the numbers.

Measured once on 2026-10-12 on an Apple M4 (10 cores, 24 GB), TensorFlow 2.16.2, Python 3.11, with the
real exported models, random-pixel inputs, 5 warm-up + 30 timed iterations per batch size, while
nothing else was deliberately running. Single runs: expect run-to-run noise, especially in p95
(a repeat of ResNet50 on CPU moved batch-8 p95 from 197 ms to 306 ms).

| Model | Device | Batch | p50 ms/batch | p95 ms/batch | img/s |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet50 | CPU | 1 / 8 / 32 | 29.9 / 167.6 / 673.4 | 33.6 / 305.8 / 736.9 | 33 / 42 / 47 |
| ResNet50 | Apple GPU (Metal) | 1 / 8 / 32 | 23.5 / 67.3 / 222.3 | 26.1 / 68.2 / 224.5 | 42 / 119 / 144 |
| ResNet101 | CPU | 1 / 8 / 32 | 60.1 / 329.8 / 1285.1 | 62.1 / 536.7 / 1457.9 | 17 / 22 / 24 |
| EfficientNetB0 | CPU | 1 / 8 / 32 | 14.4 / 67.9 / 267.7 | 16.4 / 73.0 / 385.1 | 65 / 116 / 109 |

VGG16 (541 MB) was not benchmarked. These are model-only numbers: they exclude HTTP, decoding and
resizing, and were not measured inside the Docker image. One request through the live service on the
ResNet50 model took about 1.3 s for its first call (includes graph tracing), so `latency_ms` on a cold
process is not representative; use the benchmark's warm numbers.

### Tests

```bash
pip install httpx pytest
python -m pytest
```

The tests use a tiny stub model (same `224x224x3` input, `MatchedCrop` layer and five-way softmax,
saved to a real `.keras` file and loaded through `scripts/load_model.py`), so they need neither the
real weights nor the dataset. They check the HTTP contract and the benchmark's arithmetic, not model
quality.
