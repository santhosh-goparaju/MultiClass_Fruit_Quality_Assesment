# 🍎 FruitNet: Multi-Class Fruit Quality Assessment

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**A deep learning framework that classifies fruits and their maturity stage from images for automated quality assessment.**

[Paper (CVPR Format)](#-paper) · [Results](#-results) · [Installation](#-installation) · [Usage](#-usage) · [Models](#-model-architecture)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Paper](#-paper)
- [Team](#-team)
- [License](#-license)

---

## 🔍 Overview

**FruitNet** is a deep learning-based system that performs **joint fruit type and maturity stage classification** from a single RGB image. Unlike systems that handle fruit detection or ripeness estimation separately, FruitNet combines both into a **15-class fine-grained classification** task.

### The Problem

Given an image of a fruit, the model predicts:
- 🍎 **What fruit it is** — Apple, Banana, Mango, Orange, Tomato
- 🟢 **What stage it is in** — Unripe, Ripe, Overripe

This creates **15 combined classes** (e.g., `Mango_Overripe`, `Apple_Unripe`).

### Real-World Applications

| Application | Description |
|-------------|-------------|
| 🌾 **Farm Sorting** | Automated grading at harvest |
| 🏪 **Market Quality Check** | Retail freshness inspection |
| 🏭 **Food Industry** | Supply-chain quality automation |
| 📱 **Consumer Apps** | On-device ripeness estimation |

---

## ✨ Key Features

- **Three-Stage Progressive Pipeline** — Baseline → Fine-Tuned → Improved
- **Multi-Task Y-Architecture** — Shared backbone with dual classification heads
- **Class Imbalance Handling** — Focal loss + inverse-frequency class weighting
- **Multiple Backbone Support** — VGG-16, ResNet-50, MobileNetV2, EfficientNetV2-S
- **Mixed-Precision Training** — FP16 AMP for faster training and lower memory
- **Comprehensive Evaluation** — Confusion matrices, Grad-CAM, ablation studies

---

## 🧠 Model Architecture

We employ a **Y-shaped multi-task architecture** with a shared CNN backbone branching into two parallel classification heads:

```
                    ┌─────────────────┐
                    │   Input Image   │
                    │  (224 × 224)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Shared Backbone │
                    │ ResNet-50 /     │
                    │ EfficientNetV2-S│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Global Avg    │
                    │   Pooling       │
                    └───┬─────────┬───┘
                        │         │
               ┌────────▼──┐  ┌──▼────────┐
               │ Fruit Head│  │Ripeness   │
               │ (5 cls)   │  │Head (3cls)│
               └────────┬──┘  └──┬────────┘
                        │         │
                   ┌────▼─────────▼────┐
                   │ Combined Output   │
                   │  (15 classes)     │
                   └───────────────────┘
```

### Three-Stage Training Pipeline

| Stage | Model | Description | Key Technique |
|-------|-------|-------------|---------------|
| **Stage 1** | Baseline | Feature extraction with frozen backbones | VGG-16 / ResNet-50 / MobileNetV2 (6 variants) |
| **Stage 2** | Fine-Tuned | End-to-end multi-task Y-Model | Shared ResNet-50 + dual heads (2 variants) |
| **Stage 3** | Improved | Class-imbalance-aware model | EfficientNetV2-S + Focal Loss + AMP (2 variants) |

> **Total: 10 models trained**, best model from each stage selected for final comparison.

---

## 📊 Dataset

The dataset contains images of **5 fruit types × 3 ripeness levels = 15 classes**.

| Fruit | Unripe | Ripe | Overripe |
|-------|--------|------|----------|
| 🍎 Apple | ✅ | ✅ | ✅ |
| 🍌 Banana | ✅ | ✅ | ✅ |
| 🥭 Mango | ✅ | ✅ | ✅ |
| 🍊 Orange | ✅ | ✅ | ✅ |
| 🍅 Tomato | ✅ | ✅ | ✅ |

**⚠️ Note:** The dataset exhibits class imbalance — the "ripe" category is underrepresented, which motivates our use of focal loss and class weighting.

---

## 📈 Results

### Final Model Comparison (Test Set)

| Model | Backbone | Test Accuracy | F1 (Ripe) | F1 (Macro) |
|-------|----------|:------------:|:---------:|:----------:|
| Best Baseline | ResNet-50 (Improved) | — | — | — |
| Best Fine-Tuned | ResNet-50 (Y-Model) | — | — | — |
| **Best Improved** | **EfficientNetV2-S (Y-V2)** | **95.3%** | **0.950** | **—** |

### Training Progress — Best Model (Stage 3)

| Epoch | Val Loss | Val Accuracy | F1 (Ripe) |
|:-----:|:--------:|:------------:|:---------:|
| 1 | 0.1316 | 80.5% | 0.833 |
| 5 | 0.0467 | 91.1% | 0.908 |
| 10 | 0.0359 | 93.2% | 0.932 |
| 15 | 0.0314 | 94.8% | 0.948 |
| 25 | 0.0325 | 95.1% | 0.950 |
| **30** | **0.0324** | **95.3%** | **0.950** |

---

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/santhosh-goparaju/MultiClass_Fruit_Quality_Assesment.git
cd MultiClass_Fruit_Quality_Assesment

# Install dependencies
pip install torch torchvision pandas numpy scikit-learn Pillow
```

---

## 🚀 Usage

### Training Stage 2: Y-Model (ResNet-50)

```bash
# Full training with multi-task + focal loss
python member3_train.py

# Ablation: without focal loss
python member3_train.py --no_focal

# Custom task weights (prioritise ripeness)
python member3_train.py --alpha 0.3 --beta 0.7
```

### Training Stage 3: Y-Model V2 (EfficientNetV2-S)

```bash
python member3_train_modelUpgrade.py
```

### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 32 | Training batch size |
| `--lr` | 1e-4 | Initial learning rate |
| `--alpha` | 0.5 | Fruit loss weight |
| `--beta` | 0.5 | Ripeness loss weight |
| `--unfreeze_epoch` | 10 | Epoch to unfreeze backbone |
| `--no_focal` | False | Disable focal loss (ablation) |

---

## 📁 Project Structure

```
MultiClass_Fruit_Quality_Assesment/
│
├── member3_train.py                 # Stage 2: Y-Model (ResNet-50 backbone)
├── member3_train_modelUpgrade.py    # Stage 3: Y-Model V2 (EfficientNetV2-S)
│
├── project (2).ipynb                # Experiment notebook 1
├── project (4).ipynb                # Experiment notebook 2
│
├── training_history.csv             # Stage 2 training logs (30 epochs)
├── training_history_2.csv           # Stage 3 training logs (30 epochs)
│
├── writeup/
│   ├── main.tex                     # CVPR-style LaTeX paper
│   └── overview_figure.tex          # TikZ architecture diagram
│
├── model1_Assignment1_EE655.docx    # Assignment documentation
├── Model2_Assignment1_EE655.docx    # Assignment documentation
├── cvpr-2026-submission-template.pdf
│
├── .gitignore
└── README.md
```

---

## 📝 Paper

The full project write-up is available in CVPR conference format:

- **LaTeX source:** [`writeup/main.tex`](writeup/main.tex)
- **Architecture figure:** [`writeup/overview_figure.tex`](writeup/overview_figure.tex)

### Citing this work

```bibtex
@article{fruitnet2026,
  title={FruitNet: A Deep Learning Framework for Multi-Class Fruit Quality Assessment via Fine-Grained Maturity Classification},
  author={Goparaju, Santhosh and others},
  journal={EE655 Course Project},
  year={2026}
}
```

---

## 🔧 Technical Details

### Loss Function

The combined multi-task loss:

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{fruit} + \beta \cdot \mathcal{L}_{ripeness}$$

Where $\mathcal{L}_{ripeness}$ uses **Focal Loss** with $\gamma = 2.0$:

$$FL(p_t) = -(1 - p_t)^{\gamma} \log(p_t)$$

### Training Strategy

1. **Phase 1 (Epochs 1–9):** Backbone frozen → train only classification heads
2. **Phase 2 (Epochs 10–30):** Full backbone unfrozen → fine-tune with reduced LR (cosine annealing)

### Key Improvements in Stage 3

| Improvement | Impact |
|-------------|--------|
| Focal Loss ($\gamma=2.0$) | ↑ F1 (Ripe) — focuses on hard/minority samples |
| Inverse-frequency class weights | ↑ Recall for underrepresented classes |
| EfficientNetV2-S backbone | ↑ Accuracy — better fine-texture extraction |
| Mixed-precision (FP16 AMP) | ↓ Memory ~40%, ↑ Training speed |
| Adjusted weights ($\alpha=0.3, \beta=0.7$) | ↑ Ripeness accuracy — prioritises harder task |

---

## 👥 Team

| Name | Role |
|------|------|
| **Santhosh Goparaju** | Model Development & Training |

> *EE655 — Course Project*

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repo if you find it useful!**

</div>
