# Salient Object Detection from Scratch

A from-scratch implementation of a U-Net-based Convolutional Neural Network for **Salient Object Detection (SOD)**, trained on the **DUTS dataset** (15,572 image-mask pairs). The model identifies the most visually important regions in an image and produces a binary saliency mask.

This project was built as part of an end-to-end deep learning project, covering the full pipeline from dataset preparation through training, evaluation, and live demo deployment.


## Results Summary

Best model achieves the following on the DUTS test set (2,337 unseen images):

| Metric    | Value  |
|-----------|--------|
| IoU       | 0.7762 |
| Precision | 0.8519 |
| Recall    | 0.8928 |
| F1-Score  | 0.8500 |
| MAE       | 0.0644 |

Inference time: ~70 ms per 128×128 image on CPU (~14 FPS).


## Architecture

A from-scratch U-Net with:
- **Encoder:** 4 stages, each with double Conv2D (kernel 3, padding 1) → BatchNorm → ReLU, followed by MaxPool2d(2)
- **Bottleneck:** Double Conv block at 8×8 spatial resolution + Dropout (p=0.3)
- **Decoder:** 4 stages of ConvTranspose2D (×2 upsampling) + concat skip connection + double Conv block
- **Output:** 1×1 Conv → Sigmoid → 1-channel saliency mask
- **Total parameters:** 7,763,041

**Loss:** BCE + 0.5 × (1 − soft IoU) — combines per-pixel classification with region-overlap optimization
**Optimizer:** Adam, lr=1e-3
**Training:** 20 epochs with early stopping (patience=4)


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


## Additional Analysis Performed

Beyond the core deliverables, the project also includes:

- **Statistical dataset inspection** — verified that all 15,572 image-mask pairs are readable, with 0 corrupt files and 0 fully empty masks. Computed image-size distributions (median 400×300), mask coverage (mean 23.17%, median 20.24%), and class imbalance (background-to-foreground ratio of 3.32:1). Results saved to `dataset_statistics.txt`.
- **Training curve visualization** — for each experiment, plotted train and validation loss across all 20 epochs, plus a 3-experiment comparison plot. Saved to `visualizations/training_curves_*.png`.

Both analyses informed design decisions (loss function weighting, architecture depth) and support the discussion in the project report.


## A Note for Reviewers

This repository contains all source code, training logs, evaluation results, and sample visualizations. Two large items are excluded from the repo (via `.gitignore`) and must be obtained separately:

1. **DUTS dataset (~3 GB):** Download from [the official site](http://saliencydetection.net/duts/) and place under `data/DUTS-TR/` and `data/DUTS-TE/` as shown in the project structure above. The dataset is excluded from Git because it is too large for GitHub and is owned by the original DUTS authors.

2. **Trained model checkpoints (~30 MB each):** Available upon request, or regenerate by running:
   ```bash
   python train.py baseline
   python train.py exp1_strong_aug
   python train.py exp2_deeper_lower_lr
   ```
   (Note: full training takes approximately 6 hours per experiment on CPU.) Checkpoints are excluded from Git because they exceed GitHub's recommended file size and can be reproduced by retraining.

For a quick review without setup, the `visualizations/` folder, the `*.txt` result files (`test_metrics_*.txt`, `comparison_results.txt`, `dataset_statistics.txt`), and the project report contain all key findings.


## Setup

### 1. Install Python 3.9 or newer

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure paths (Windows-specific paths in code)

The code uses absolute paths configured for the original development environment. To run on your machine, update these constants at the top of each file:

- `BASE_DIR` in `train.py`, `evaluate.py`, and `data_loader.py` — points to the project folder
- `CKPT_DIR` in `train.py`, `evaluate.py`, and `app.py` — points to where you want model checkpoints stored

### 4. Download the DUTS dataset

Download DUTS from the [official site](http://saliencydetection.net/duts/) and place it under `data/`:

```
data/
├── DUTS-TR/
│   ├── DUTS-TR-Image/    (10,553 jpg images)
│   └── DUTS-TR-Mask/     (10,553 png masks)
└── DUTS-TE/
    ├── DUTS-TE-Image/    (5,019 jpg images)
    └── DUTS-TE-Mask/     (5,019 png masks)
```

### 5. Generate train/val/test splits

```bash
python data_loader.py
```

This creates `data/splits/train.txt`, `val.txt`, `test.txt` with a deterministic 70/15/15 split (seed=42).


## Usage

### Train

```bash
python train.py baseline               
python train.py exp1_strong_aug            
python train.py exp2_deeper_lower_lr      
python train.py baseline --fresh           
```

Training automatically resumes from the last checkpoint if one exists.


### Evaluate on test set

```bash
python evaluate.py metrics baseline              
python evaluate.py visualize baseline              
python evaluate.py compare                        
python evaluate.py verify_sklearn baseline         
```


### Run the live demo

```bash
python -m streamlit run app.py
```

Opens a browser interface at http://localhost:8501. Upload any image to see the predicted saliency mask, probability map, overlay, and inference time.


## Experiments

Three configurations were trained for 20 epochs on CPU:

| Experiment                 | Depth | Augmentation | IoU Weight | Learning Rate | Best Val Loss |
|----------------------------|-------|--------------|------------|----------------|----------------|
| Baseline                   | 4     | basic        | 0.5        | 1e-3           | 0.2905         |
| Exp1 (strong augmentation) | 4     | strong       | 1.0        | 1e-3           | 0.4451         |
| Exp2 (deeper + lower LR)   | 5     | basic        | 0.5        | 5e-4           | 0.2919         |

Final test-set comparison:

| Experiment            | IoU    | Precision | Recall | F1     | MAE    |
|-----------------------|--------|-----------|--------|--------|--------|
| baseline              | 0.7762 | 0.8519    | 0.8928 | 0.8500 | 0.0644 |
| exp1_strong_aug       | 0.7661 | 0.8504    | 0.8797 | 0.8426 | 0.0651 |
| exp2_deeper_lower_lr  | 0.7742 | 0.8530    | 0.8846 | 0.8482 | 0.0681 |

The baseline configuration emerged as the strongest within the 20-epoch CPU training budget.


## Key Design Decisions

- **U-Net with 4 levels:** Standard architecture for image-to-image tasks; balances spatial detail and receptive field.
- **BCE + IoU loss:** BCE provides per-pixel gradient signal; IoU loss directly optimizes the region-overlap metric.
- **128×128 resolution:** Trades fine detail for tractable CPU training time.
- **Per-image-then-averaged metrics:** Standard convention in SOD literature; cross-verified against scikit-learn (see `verify_with_sklearn` in `evaluate.py`).


## Tools and Libraries

- Python 3.9+
- PyTorch 2.x
- NumPy
- OpenCV (image inspection)
- Matplotlib (training curves and visualizations)
- scikit-learn (metric verification)
- tqdm (progress bars)
- Streamlit (live demo)
- Pillow (image I/O)


## Limitations and Future Work

- **Multi-subject scenes:** When two salient objects are spatially adjacent, the model tends to merge them or pick only one. Multi-scale fusion architectures (BASNet, HED) would help.
- **Small objects:** Subjects covering <5% of the image are sometimes missed entirely.
- **Bright text/signs:** The model can incorrectly flag attention-grabbing background elements as salient.
- **CPU-only training:** With GPU access, longer training runs with deeper architectures could likely surpass current results.


## Author

Dafina Keqmezi
