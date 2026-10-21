---
title: ML Capstone Studio - Regression, Classification & Clustering
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app/app.py
pinned: false
---

# 🚀 ML Capstone Studio: End-to-End Machine Learning Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-blue?style=for-the-badge&logo=render)](https://ml-capstone-studio.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Framework](https://img.shields.io/badge/UI-Gradio%206-orange?style=for-the-badge)](https://gradio.app)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

> 🌐 **Live Web Application Demo:**  
> 👉 **[https://ml-capstone-studio.onrender.com](https://ml-capstone-studio.onrender.com)**

---

## 📌 Project Overview

This repository houses the complete deliverables for the **Machine Learning Capstone Project (Reviews 1 & 2)**. It provides end-to-end pipelines, reproducible benchmarks, serialized models, and a sleek SaaS-style interactive web application spanning all three foundational ML paradigms: **Regression** (10 algorithms on Flavors of Cacao), **Classification** (10 algorithms Part A & B on Mobile Price), and **Clustering** (K-Means & Agglomerative on UCI Multiple Features).

---

## 🖥️ Interactive Web Application Features

The interactive web application (`app/app.py`) provides an executive-grade interface with 5 dedicated workspaces:

1. **Executive Dashboard**: High-level system overview, KPI metric cards, dataset summaries, and end-to-end architecture pipeline diagrams.
2. **Cocoa Rating Regressor (Review 1)**: Interactive chocolate rating estimator powered by the top-performing regression models with dynamic feature inputs (Cocoa %, Review Date, Bean Origin, Company Location) and real-time inference.
3. **Mobile Price Classifier (Review 2)**: Smartphone price band predictor (Low, Medium, High, Very High) with interactive spec sliders, preset hardware profiles, real-time class probability distributions, and feature impact analysis.
4. **Digit Clustering Studio (Review 2)**: Interactive 2D PCA and t-SNE scatter plots across all algorithms and linkages, cluster metric cards, live 15×16 pixel handwritten digit image reconstruction, and custom data point assignment.
5. **Model Benchmark Lab**: Cross-track comparison leaderboards, comprehensive evaluation metrics, radar comparison charts, and interactive confusion matrices.

---

## 📊 Datasets & Problem Formulations

### 1. Regression: Flavors of Cacao
- **Records**: 1,795 expert chocolate ratings.
- **Features**: Cocoa percentage, review date, bean origin country, company location, and bean type.
- **Target**: `Rating` (continuous scale from 1.0 to 5.0).

### 2. Classification: Mobile Price Classification
- **Records**: 2,000 mobile phones with 20 numeric technical features (battery capacity, RAM, cameras, screen resolution, weight, cores, and connectivity).
- **Target**: `price_range` with four balanced classes (500 samples each):
  - `0`: Low Cost
  - `1`: Medium Cost
  - `2`: High Cost
  - `3`: Very High Cost

### 3. Clustering: UCI Multiple Features (mfeat)
- **Records**: 2,000 handwritten digits (0–9, 200 samples per digit) extracted from Dutch utility maps, without label supervision during clustering.
- **Combined Dimensions**: 649 continuous features concatenated from 6 UCI representations:
  - `mfeat-fou` (76): Fourier coefficients of character shapes
  - `mfeat-fac` (216): Profile correlations
  - `mfeat-kar` (64): Karhunen-Loève coefficients
  - `mfeat-pix` (240): 2×3 window pixel averages (reconstructs 15×16 digit bitmaps)
  - `mfeat-zer` (47): Zernike moments
  - `mfeat-mor` (6): Morphological characteristics

---

## 📈 Benchmark Results

### 1. Regression Track (10-Algorithm Comparison)

| Rank | Algorithm | $R^2$ Score | RMSE | MAE |
|:---:|:---|---:|---:|---:|
| **1** | **Ridge Regression** | **0.9999** | **0.0035** | **0.0018** |
| 2 | Linear Regression | 0.9999 | 0.0035 | 0.0018 |
| 3 | Lasso Regression | 0.9996 | 0.0092 | 0.0075 |
| 4 | Decision Tree Regressor | 0.9995 | 0.0097 | 0.0009 |
| 5 | Random Forest Regressor | 0.9993 | 0.0117 | 0.0018 |
| 6 | Gradient Boosting Regressor | 0.9992 | 0.0128 | 0.0026 |
| 7 | Polynomial Regression | 0.9829 | 0.0585 | 0.0132 |
| 8 | Support Vector Regressor (SVR) | 0.8840 | 0.1526 | 0.0969 |
| 9 | K-Nearest Neighbors Regressor | 0.3945 | 0.3485 | 0.2682 |
| 10 | ElasticNet Regression | -0.0004 | 0.4479 | 0.3638 |

---

### 2. Classification Track (Consolidated 10-Algorithm Comparison, Test Set $n=400$)

All models evaluate on an identical stratified 80/20 train/test split (`random_state=42`) with leakage-free pipeline scaling:

| Model | Track / Part | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | OvR ROC-AUC |
|:---|:---:|---:|---:|---:|---:|---:|
| **SVM (SVC - Tuned)** | **A (Review 1)** | **0.9750** | **0.9750** | **0.9750** | **0.9750** | **0.9994** |
| Logistic Regression | A (Review 1) | 0.9825 | 0.9826 | 0.9825 | 0.9825 | 0.9991 |
| MLP (Neural Network) | B (Review 2) | 0.9575 | 0.9576 | 0.9575 | 0.9575 | 0.9980 |
| AdaBoost | B (Review 2) | 0.9375 | 0.9383 | 0.9375 | 0.9377 | 0.9933 |
| Gradient Boosting | B (Review 2) | 0.9375 | 0.9374 | 0.9375 | 0.9374 | 0.9923 |
| Bagging (Decision Tree) | B (Review 2) | 0.8975 | 0.8978 | 0.8975 | 0.8976 | 0.9877 |
| Random Forest | B (Review 2) | 0.8875 | 0.8876 | 0.8875 | 0.8872 | 0.9804 |
| Decision Tree | A (Review 1) | 0.8375 | 0.8367 | 0.8375 | 0.8370 | 0.9511 |
| Gaussian Naive Bayes | A (Review 1) | 0.8100 | 0.8113 | 0.8100 | 0.8105 | 0.9506 |
| K-Nearest Neighbors | A (Review 1) | 0.7200 | 0.7219 | 0.7200 | 0.7203 | 0.8792 |

> **Champion Model:** Support Vector Classifier (`SVC(C=10, kernel='linear')`), achieving **97.50% Test Accuracy**, **0.9750 Weighted F1**, and **0.9994 ROC-AUC**.

<details>
<summary>🔍 Click to view 5-fold Cross-Validation Hyperparameter Tuning Details</summary>

| Model | Tuned Hyperparameters | CV F1 (5-Fold) | Baseline F1 | Tuned Test F1 |
|:---|:---|---:|---:|---:|
| **SVM (SVC)** | `C=10, kernel=linear` | **0.9650** | 0.8956 | **0.9750** |
| Logistic Regression | `C=100` | 0.9632 | 0.9650 | 0.9825 |
| MLP (Neural Net) | `activation=tanh, hidden_layer_sizes=(128, 64)` | 0.9549 | 0.9275 | 0.9575 |
| AdaBoost | `estimator=DecisionTree(max_depth=3), learning_rate=0.5, n_estimators=200` | 0.9199 | 0.4437 | 0.9377 |
| Gradient Boosting | `learning_rate=0.1, max_depth=2, n_estimators=300` | 0.9034 | 0.9123 | 0.9374 |
| Bagging | `max_features=1.0, max_samples=1.0, n_estimators=100` | 0.8868 | 0.8976 | 0.8976 |
| Random Forest | `max_depth=10, n_estimators=300` | 0.8728 | 0.8797 | 0.8872 |
| Decision Tree | `max_depth=8, min_samples_leaf=10` | 0.8372 | 0.8302 | 0.8370 |
| Naive Bayes | `var_smoothing=1e-11` | 0.8044 | 0.8105 | 0.8105 |
| KNN | `metric=manhattan, n_neighbors=31` | 0.6639 | 0.5054 | 0.7203 |

</details>

---

### 3. Clustering Track ($k = 10$)

Clustering evaluated across intrinsic validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and external validation against ground-truth digits (ARI, NMI):

| Algorithm / Linkage | Silhouette Score $\uparrow$ | Davies-Bouldin $\downarrow$ | Calinski-Harabasz $\uparrow$ | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) |
|:---|---:|---:|---:|---:|---:|
| **Agglomerative (Ward)** | **0.1286** | 2.2943 | 139.36 | **0.9026** | **0.9116** |
| **K-Means** | **0.1325** | **2.2376** | **145.29** | **0.8376** | **0.8569** |
| Agglomerative (Average) | 0.0888 | 1.5535 | 61.04 | 0.3095 | 0.5774 |
| Agglomerative (Complete) | 0.0660 | 2.5599 | 86.94 | 0.2962 | 0.4914 |
| Agglomerative (Single) | -0.0160 | 0.8118 | 1.56 | 0.0000 | 0.0089 |

---

## 📂 Repository Structure

```
├── README.md                     # Comprehensive project documentation
├── requirements.txt              # Core production & training dependencies
├── render.yaml                   # Infrastructure-as-code for Render deployment
├── data/
│   ├── flavors_of_cacao.csv      # Review 1: Chocolate Bar Ratings dataset
│   ├── train.csv                 # Review 2: Mobile Price Classification dataset
│   └── mfeat/                    # Review 2: UCI Multiple Features (6 modality files)
├── notebooks/
│   ├── regression.ipynb          # Review 1: 10 Regression models & EDA
│   ├── classification.ipynb      # Review 2: Classification Part A + B & GridSearch
│   └── clustering.ipynb          # Review 2: K-Means, Hierarchical, PCA & t-SNE
├── src/                          # Modular engineering modules
│   ├── config.py                 # Seeds, file paths, global parameters
│   ├── data_loader.py            # Clean loaders with binary caching (.npz)
│   ├── regression.py             # 10 Regression model pipelines & serialization
│   ├── classification.py         # 10 Classification pipelines & GridSearch
│   ├── clustering.py             # Elbow analysis, K-Means & Dendrogram generation
│   └── visualize.py              # Exportable Plotly & Matplotlib charts
├── train_pipeline.py             # End-to-end training and serialization runner
├── models/                       # Serialized production artifacts (.joblib)
│   ├── best_regressor.joblib     # Serialized champion regression model
│   ├── best_classifier.joblib    # Serialized champion classification model
│   └── clustering.joblib         # Serialized K-Means & scaler models
├── reports/                      # Benchmark tables (.csv) & evaluation outputs
└── app/
    ├── app.py                    # Multi-tab Gradio SaaS Web Application
    └── ui/
        ├── components.py         # Reusable chart generators & SVG digit visualizer
        └── style.css             # High-contrast modern SaaS design system
```

---

## ⚡ Quick Start & Local Execution

### 1. Environment Setup
Python 3.11 or 3.12 is recommended:

```bash
# Clone the repository
git clone https://github.com/tharun-0106/ML_Capstone_Project.git
cd ML_Capstone_Project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Web UI
Launch the local web server:
```bash
python app/app.py
```
Open your browser and navigate to **`http://localhost:7860`**.

### 3. (Optional) Re-run the Entire Training Pipeline
```bash
python train_pipeline.py
```

---

## 👥 Authors & Academic Context

- **Course:** 23CSE301 - Machine Learning
- **Institution:** Department of Computer Science & Engineering
- **Project Scope:** Capstone Review 1 (Regression) & Review 2 (Classification & Clustering)
