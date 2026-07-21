# Transfer Learning for Image Classification

Final project for the MLDS program. Classifies images across five classes using four ImageNet-pretrained convolutional backbones (frozen) with a small regularized classification head trained on top.

## Contents

- `notebook/` — main analysis notebook (`Bharadwaj_Sujendra_Giri_Prasad_Final_Project.ipynb`)
- `scripts/` — helper modules for loading pretrained backbones (`load_model.py`) and defining the classification head (`model_layers.py`)

## Setup

The notebook expects a Python environment with PyTorch/TensorFlow (see imports in the notebook) and the project's `data/` directory populated with the supplied image dataset. Data files are not included in this repository due to size; place them under `data/` before running.

## Author

Sujendra Giri Prasad Bharadwaj
