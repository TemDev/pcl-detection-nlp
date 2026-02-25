# PCL Detection using RoBERTa

## Repository Structure

PCL-DETECTION-NLP/

code/

analysis.ipynb # EDA, threshold tuning, error analysis
model.ipynb # Model training & prediction pipeline

data/ # Dataset files

dev.txt # Dev predictions (BestModel)
test.txt # Test predictions (BestModel)

README.md

## Project Overview

This project implements a binary classifier for detecting **Patronising and Condescending Language (PCL)** using the *Don't Patronize Me!* dataset (SemEval-2022 Task 4, Subtask 1).

The official shared-task baseline:

- **RoBERTa-base**
- Dev F1 (PCL): **0.48**

Our final submitted model (**BestModel**) achieved:

- **Dev F1 (PCL): 0.5779**
- Precision (PCL): 0.4879  
- Recall (PCL): 0.7085  
- Accuracy: 0.9016  
- Optimal decision threshold: **0.55**

Evaluation follows the official metric: **F1 score of the positive (PCL) class**.

---

## What Was Implemented

### 1. Label Mapping

Original 5-class labels (0–4) were converted to binary:

- {0,1} → 0 (No PCL)
- {2,3,4} → 1 (PCL)

This follows the official shared-task grouping.

---

### 2. Handling Class Imbalance

The dataset is highly imbalanced (~10% positive).

Instead of downsampling:

- Used full dataset
- Applied **class-weighted cross-entropy loss**
- Optimised for F1 of positive class

---

### 3. Model

Base model: roberta-base


Fine-tuned with:

- Learning rate: 2e-5
- Max length: 256
- Early stopping
- Weighted loss
- Threshold tuning (instead of fixed 0.5)

---

### 4. Threshold Optimisation

Since the official metric is F1(PCL), the classification threshold was tuned on the dev set.

Best threshold: 0.55


This significantly improved minority-class recall.

---

### 5. Error Analysis

Performed in `analysis.ipynb`:

- Confusion matrix
- Precision–Recall curve (AP = 0.6504)
- Probability distribution analysis
- Manual qualitative error categorisation
- Lexical shortcut bias investigation

Key finding:

False positive rate in high-signal lexical subset:
- 0.40 (inside subset)
- 0.075 (outside subset)

This suggests the model partially relies on vulnerability-related lexical shortcuts.

---

## Running the Notebooks

⚠️ **Important: Kaggle Environment**

Experiments were run on **Kaggle Notebooks**.

The file paths inside `model.ipynb` use Kaggle-style paths such as:

```python
/kaggle/input/...


