# Breast Cancer Classification with Deep Learning and Optimization
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview
This repository contains the implementation of a **Breast Cancer Classification** framework using deep learning and optimization techniques. The proposed model integrates **a 32-layer CNN inspired by YOLO, U-Net, and ResNet** and utilizes **Modified Grey Wolf Optimization (mGWO)** to refine feature selection, reduce redundancy, and improve classification accuracy.

The model is trained and evaluated on three benchmark mammogram datasets: **MIAS, DDSM, and INbreast**. The implementation includes **data preprocessing, augmentation, model training, optimization, and evaluation**.

---

## Features
✅ **Haze-Removed Adaptive Technique (HRAT)** for contrast enhancement.  
✅ **Data augmentation** (rotation, flipping, noise addition) to balance datasets.  
✅ **32-layer CNN architecture** with residual learning blocks and multi-scale convolutions.  
✅ **Modified Grey Wolf Optimization (mGWO)** for feature selection and refinement.  
✅ **Comprehensive model evaluation** with accuracy, sensitivity, specificity, F1-score, and kappa.  
✅ **Support for MIAS, DDSM, and INbreast datasets**.  

---

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/imashoodnasir/Breast-Cancer-Classification.git
cd Breast-Cancer-Classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation
1. **Download the Datasets**  
   - MIAS: [Download Link](http://peipa.essex.ac.uk/pix/mias/)  
   - DDSM: [Download Link](https://www.kaggle.com/datasets/kmader/mias-mammography)  
   - INbreast: [Download Link](https://medical-datasets.com/inbreast/)  

2. **Organize the Dataset Directory Structure**
   ```
   dataset/
   ├── MIAS/
   │   ├── benign/
   │   ├── malignant/
   │   ├── normal/
   ├── DDSM/
   │   ├── benign/
   │   ├── malignant/
   ├── INbreast/
       ├── benign/
       ├── malignant/
   ```

3. **Preprocess & Augment the Data**  
   Run the preprocessing script to apply **HRAT, cropping, normalization, and augmentation**:
   ```bash
   python preprocess.py
   ```

---

## Usage
### 1. Train the Model
```bash
python train.py --epochs 150 --batch_size 32 --learning_rate 0.001
```

### 2. Evaluate the Model
```bash
python evaluate.py
```

### 3. Feature Selection using mGWO
```bash
python optimize.py
```

---

## Model Architecture
- **Input Size:** `224 × 224 × 3`
- **Feature Extraction:** Multi-scale convolutions (3×3, 5×5, 7×7)
- **Residual Learning Blocks:** Prevent vanishing gradients
- **Strided Convolutions:** Instead of max pooling for downsampling
- **Final Layers:** Global Average Pooling → Dropout (0.3) → Dense Layer

---

## Performance
| Dataset  | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-score | Kappa |
|----------|------------|----------------|----------------|---------|-------|
| **MIAS**    | 97.74      | 98.02          | 98.47          | 98.16   | 99.15 |
| **DDSM**    | 99.37      | 97.98          | 96.88          | 99.05   | 97.36 |
| **INbreast**| 99.02      | 99.86          | 97.97          | 98.16   | 98.97 |

---

## Results Visualization
### ROC Curves
Run the script to generate **Receiver Operating Characteristic (ROC) curves**:
```bash
python visualize_results.py
```

### Grad-CAM and LIME Interpretability
To visualize model decisions, use:
```bash
python explainability.py
```

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contact
For queries, reach out via [GitHub Issues](https://github.com/imashoodnasir/Breast-Cancer-Classification/issues). 🚀
