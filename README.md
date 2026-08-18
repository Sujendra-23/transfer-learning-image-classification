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
