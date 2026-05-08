# Salient Object Detection rom Scratch (DUTS)

A from-scratch implementation of a U-Net-based Convolutional Neural Network for **Salient Object Detection (SOD)**, trained on the **DUTS dataset** (15,572 image-mask pairs). The model identifies the most visually important regions in an image and produces a binary saliency mask.

This project covers the full deep learning pipeline: dataset preparation, model design, training with multiple experiment configurations, quantitative evaluation, and an interactive demo deployment.

---

## Table of Contents

- [Results](#results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Usage](#usage)
- [Experiments](#experiments)
- [Additional Analysis](#additional-analysis)
- [Key Design Decisions](#key-design-decisions)
- [Tools and Libraries](#tools-and-libraries)
- [Limitations and Future Work](#limitations-and-future-work)
- [Notes for Reviewers](#notes-for-reviewers)
- [Author](#author)

---

## Results

The best-performing model achieves the following on the DUTS test set (2,337 unseen images):

| Metric    | Value  |
|-----------|--------|
| IoU       | 0.7762 |
| Precision | 0.8519 |
| Recall    | 0.8928 |
| F1-Score  | 0.8500 |
| MAE       | 0.0644 |

**Inference time:** ~70 ms per 128×128 image on CPU (~14 FPS).

---

## Architecture

A from-scratch U-Net with the following components:

- **Encoder:** 4 stages, each with double Conv2D (kernel 3, padding 1) → BatchNorm → ReLU, followed by MaxPool2d(2)
- **Bottleneck:** Double Conv block at 8×8 spatial resolution + Dropout (p=0.3)
- **Decoder:** 4 stages of ConvTranspose2D (×2 upsampling) + concatenated skip connection + double Conv block
- **Output:** 1×1 Conv → Sigmoid → 1-channel saliency mask
- **Total parameters:** 7,763,041

**Loss:** `BCE + 0.5 × (1 − soft IoU)` — combines per-pixel classification with region-overlap optimization
**Optimizer:** Adam (lr=1e-3)
**Training:** 20 epochs with early stopping (patience=4)

---

## Project Structure

```
ML_DAFINA_AI/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                          
├── data_loader.py                 
├── sod_model.py                  
├── train.py                       
├── evaluate.py                     
├── data/
│   ├── DUTS-TE/
│   │   ├── DUTS-TE-Image/
│   │   └── DUTS-TE-Mask/
│   ├── DUTS-TR/
│   │   ├── DUTS-TR-Image/
│   │   └── DUTS-TR-Mask/
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── logs/
│   ├── baseline.txt
│   ├── exp1_strong_aug.txt
│   └── exp2_deeper_lower_lr.txt
├── visualizations/
├── dataset_statistics.txt
├── test_metrics_baseline.txt
├── test_metrics_exp1_strong_aug.txt
├── test_metrics_exp2_deeper_lower_lr.txt
├── comparison_results.txt
└── comparison_results.csv
```

---

## Quick Start

```bash
git clone https://github.com/dafinaak/SOD_Project_ML_DafinaKeqmezi.git
cd SOD_Project_ML_DafinaKeqmezi

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Launch the live demo
python -m streamlit run app.py
```

The demo opens at [http://localhost:8501](http://localhost:8501). To train or evaluate the model, follow the [Detailed Setup](#detailed-setup) section below.

---

## Detailed Setup

### 1. Prerequisites

- **Python 3.9 or newer** ([download](https://www.python.org/downloads/))
- **Git** ([download](https://git-scm.com/downloads))
- ~3 GB free disk space for the dataset

### 2. Clone the Repository

```bash
git clone https://github.com/dafinaak/SOD_Project_ML_DafinaKeqmezi.git
cd SOD_Project_ML_DafinaKeqmezi
```

### 3. Create a Virtual Environment (recommended)

```bash
python -m venv venv
```

Activate it:

| Platform        | Command                  |
|-----------------|--------------------------|
| Windows (PS)    | `venv\Scripts\Activate.ps1` |
| Windows (CMD)   | `venv\Scripts\activate.bat` |
| macOS / Linux   | `source venv/bin/activate`  |

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Configure Project Paths

The code uses absolute paths configured for the original development environment. Update the following constants at the top of each file to match your local setup:

- `BASE_DIR` in `train.py`, `evaluate.py`, and `data_loader.py` — points to the project root folder
- `CKPT_DIR` in `train.py`, `evaluate.py`, and `app.py` — points to where model checkpoints will be stored

### 6. Download the DUTS Dataset

Download the DUTS dataset from the [official site](http://saliencydetection.net/duts/) and extract it under `data/`:

```
data/
├── DUTS-TR/
│   ├── DUTS-TR-Image/    (10,553 jpg images)
│   └── DUTS-TR-Mask/     (10,553 png masks)
└── DUTS-TE/
    ├── DUTS-TE-Image/    (5,019 jpg images)
    └── DUTS-TE-Mask/     (5,019 png masks)
```

> **Note:** The dataset is excluded from the repository (~3 GB) and must be obtained separately.

### 7. Generate Train/Val/Test Splits

```bash
python data_loader.py
```

Creates `data/splits/train.txt`, `val.txt`, and `test.txt` using a deterministic 70/15/15 split (seed=42).

---

## Usage

### Training

```bash
python train.py baseline                  
python train.py exp1_strong_aug          
python train.py exp2_deeper_lower_lr      
python train.py baseline --fresh          
```

Training automatically resumes from the latest checkpoint if one exists.

### Evaluation

```bash
python evaluate.py metrics baseline          
python evaluate.py visualize baseline        
python evaluate.py compare                    
python evaluate.py verify_sklearn baseline    
```

### Live Demo

```bash
python -m streamlit run app.py
```

Opens an interactive browser interface at [http://localhost:8501](http://localhost:8501). Upload any image to see the predicted saliency mask, probability map, overlay, and inference time.

---

## Experiments

Three configurations were trained for 20 epochs on CPU:

| Experiment                 | Depth | Augmentation | IoU Weight | Learning Rate | Best Val Loss |
|----------------------------|-------|--------------|------------|---------------|---------------|
| Baseline                   | 4     | basic        | 0.5        | 1e-3          | 0.2905        |
| Exp1 (strong augmentation) | 4     | strong       | 1.0        | 1e-3          | 0.4451        |
| Exp2 (deeper + lower LR)   | 5     | basic        | 0.5        | 5e-4          | 0.2919        |

**Final test-set comparison:**

| Experiment            | IoU    | Precision | Recall | F1     | MAE    |
|-----------------------|--------|-----------|--------|--------|--------|
| baseline              | 0.7762 | 0.8519    | 0.8928 | 0.8500 | 0.0644 |
| exp1_strong_aug       | 0.7661 | 0.8504    | 0.8797 | 0.8426 | 0.0651 |
| exp2_deeper_lower_lr  | 0.7742 | 0.8530    | 0.8846 | 0.8482 | 0.0681 |

The **baseline** configuration emerged as the strongest within the 20-epoch CPU training budget.

---

## Additional Analysis

Beyond the core deliverables, the project also includes:

- **Statistical dataset inspection** — verified that all 15,572 image-mask pairs are readable, with 0 corrupt files and 0 fully empty masks. Computed image-size distributions (median 400×300), mask coverage (mean 23.17%, median 20.24%), and class imbalance (background-to-foreground ratio of 3.32:1). Results saved to `dataset_statistics.txt`.
- **Training curve visualization** — for each experiment, plotted train and validation loss across all 20 epochs, plus a 3-experiment comparison plot. Saved to `visualizations/training_curves_*.png`.

Both analyses informed design decisions (loss function weighting, architecture depth) and support the discussion in the project report.

---

## Key Design Decisions

- **U-Net with 4 levels:** Standard architecture for image-to-image tasks; balances spatial detail and receptive field.
- **BCE + IoU loss:** BCE provides a per-pixel gradient signal; IoU loss directly optimizes the region-overlap metric.
- **128×128 resolution:** Trades fine detail for tractable CPU training time.
- **Per-image-then-averaged metrics:** Standard convention in SOD literature; cross-verified against scikit-learn (see `verify_with_sklearn` in `evaluate.py`).

---

## Tools and Libraries

| Category          | Library                                |
|-------------------|----------------------------------------|
| Language          | Python 3.9+                            |
| Deep Learning     | PyTorch 2.x, torchvision               |
| Numerical / Image | NumPy, OpenCV, Pillow                  |
| Visualization     | Matplotlib                             |
| Evaluation        | scikit-learn                           |
| UI / Demo         | Streamlit                              |
| Utilities         | tqdm                                   |

---

## Limitations and Future Work

- **Multi-subject scenes:** When two salient objects are spatially adjacent, the model tends to merge them or pick only one. Multi-scale fusion architectures (BASNet, HED) would help.
- **Small objects:** Subjects covering <5% of the image are sometimes missed entirely.
- **Bright text/signs:** The model can incorrectly flag attention-grabbing background elements as salient.
- **CPU-only training:** With GPU access, longer training runs with deeper architectures could likely surpass the current results.

---

## Notes for Reviewers

This repository contains all source code, training logs, evaluation results, and sample visualizations. Two large items are excluded from the repo (via `.gitignore`) and must be obtained separately:

1. **DUTS dataset (~3 GB):** Download from the [official site](http://saliencydetection.net/duts/) and place under `data/DUTS-TR/` and `data/DUTS-TE/` as shown in the [project structure](#project-structure). The dataset is excluded from Git because it is too large for GitHub and is owned by the original DUTS authors.

2. **Trained model checkpoints (~30 MB each):** Available upon request, or regenerate by running:
   ```bash
   python train.py baseline
   python train.py exp1_strong_aug
   python train.py exp2_deeper_lower_lr
   ```
   Full training takes approximately 6 hours per experiment on CPU. Checkpoints are excluded from Git because they exceed GitHub's recommended file size and can be reproduced by retraining.

For a quick review without setup, the `visualizations/` folder, the `*.txt` result files (`test_metrics_*.txt`, `comparison_results.txt`, `dataset_statistics.txt`), and the project report contain all key findings.

---

## Author

**Dafina Keqmezi**
