#  Land Use Classification — EuroSAT / Sentinel-2

> Deep learning pipeline for satellite image classification using custom CNN and transfer learning (EfficientNetB3, MobileNetV2), with Grad-CAM explainability.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://kaggle.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This project tackles **Land Use and Land Cover (LULC) classification** using the [EuroSAT](https://github.com/phelber/EuroSAT) dataset — 27,000 labeled RGB satellite images (64×64 px) from Sentinel-2, covering 10 European land use classes.

The full pipeline includes:
- Custom CNN baseline trained from scratch
- Transfer learning with **EfficientNetB3** and **MobileNetV2** (two-stage: freeze → fine-tune)
- **Grad-CAM** visualizations for model explainability
- Comprehensive evaluation: confusion matrices, per-class F1, radar charts

---

##  Dataset — EuroSAT

| Property | Value |
|---|---|
| Total images | 27,000 |
| Image size | 64 × 64 px (RGB) |
| Number of classes | 10 |
| Source | Sentinel-2 satellite (ESA) |

**Classes:**

| Class | Description |
|---|---|
| AnnualCrop | Annual agricultural crops |
| Forest | Dense forest areas |
| HerbaceousVegetation | Grasslands and meadows |
| Highway | Major road infrastructure |
| Industrial | Industrial zones |
| Pasture | Grazing land |
| PermanentCrop | Orchards, vineyards |
| Residential | Urban residential zones |
| River | Rivers and waterways |
| SeaLake | Seas, lakes, and water bodies |

Dataset available on Kaggle: [sentinel-2-satellite-imagery](https://www.kaggle.com/datasets/gallo33henrique/sentinel-2-satellite-imagery)

---

##  Project Structure

```
satellite-land-classification/
│
├── satellite-imagery.ipynb     # Main notebook (full pipeline)
├── README.md
└── .gitignore
```

---

##  Pipeline

```
Raw images (64×64 RGB)
       │
       ▼
Train / Val / Test split (70% / 15% / 15%, stratified)
       │
       ▼
tf.data pipeline
  ├── Baseline CNN  → [0, 1] normalized
  └── TL models     → [0, 255] raw + internal preprocess_input()
       │
       ▼
Data augmentation (flip, rotate, zoom, brightness, contrast)
       │
       ▼
Model training
  ├── Baseline CNN (custom, 4 conv blocks)
  ├── EfficientNetB3 (ImageNet, freeze → fine-tune top 30 layers)
  └── MobileNetV2   (ImageNet, freeze → fine-tune top 30 layers)
       │
       ▼
Evaluation (accuracy, precision, recall, F1, confusion matrix)
       │
       ▼
Grad-CAM explainability
```

---

##  Results

> Exact numbers depend on your training run. Typical results on EuroSAT:

| Model | Test Accuracy | F1-Score | Parameters |
|---|---|---|---|
| Baseline CNN | ~75–82% | ~0.75–0.82 | ~2M |
| MobileNetV2 | ~88–92% | ~0.88–0.92 | ~3.5M |
| EfficientNetB3 | ~91–95% | ~0.91–0.95 | ~12M |

---

## Grad-CAM Explainability

Grad-CAM highlights which regions of an image the model focuses on when making a prediction. This confirms the model attends to:
- **Road textures** → Highway
- **Tree canopies** → Forest
- **Water surfaces** → River / SeaLake
- **Building patterns** → Residential / Industrial

---

##  How to Run

### On Kaggle (recommended)

1. Go to [Kaggle](https://kaggle.com) and open a new notebook
2. Add the dataset: `sentinel-2-satellite-imagery`
3. Upload `satellite-imagery.ipynb`
4. Enable GPU: **Settings → Accelerator → GPU T4**
5. Run all cells

### Locally

```bash
git clone https://github.com/YOUR_USERNAME/satellite-land-classification.git
cd satellite-land-classification

pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow

jupyter notebook satellite-imagery.ipynb
```

>  The notebook uses mixed precision (`float16`) and GPU memory growth — recommended to run on GPU.

---

##  Predict on a Custom Image

```python
from PIL import Image
import numpy as np

image_path = "path/to/your/image.jpg"
img = Image.open(image_path).convert("RGB").resize((64, 64))

# For EfficientNetB3 / MobileNetV2 — raw [0, 255], NO division by 255
img_array = np.array(img).astype("float32")
img_tensor = np.expand_dims(img_array, axis=0)

predictions = eff_model.predict(img_tensor, verbose=0)[0]
top3 = np.argsort(predictions)[::-1][:3]

for i, idx in enumerate(top3):
    print(f"{i+1}. {CLASS_NAMES[idx]}: {predictions[idx]*100:.2f}%")
```

>  **Important:** Transfer learning models (`eff_model`, `mob_model`) expect raw `[0, 255]` images. The baseline CNN expects `[0, 1]`. Using the wrong range causes wrong predictions (typically always predicts SeaLake).

---

## 🔧 Key Technical Notes

| Topic | Detail |
|---|---|
| Mixed precision | `float16` training, `float32` output layer |
| Preprocessing | TL models use `preprocess_input()` internally — do NOT normalize to [0,1] before passing |
| Grad-CAM | Searches for Conv2D inside sub-models (EfficientNet/MobileNet nest layers) |
| Fine-tuning | Last 30 layers unfrozen, LR = 1e-5 (EfficientNet) / 5e-6 (MobileNet) |
| Augmentation | Flip, rotate ±20°, zoom ±15%, brightness ±15%, contrast ±20% |

---

##  Dependencies

```
tensorflow >= 2.12
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
pillow
```

---

##  References

- Helber et al. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV.
- Howard et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.*

---

##  License

MIT License — free to use, modify, and distribute.

---

<p align="center">Made with ❤️ using TensorFlow, Keras, and the EuroSAT dataset</p>
