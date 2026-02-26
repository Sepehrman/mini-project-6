# Mini Project 6: Transfer Learning Application

**COMP 9130 — Applied Artificial Intelligence**

A transfer learning project applying **feature extraction** and **fine-tuning** to flower species classification using the Oxford 102 Flowers dataset. Both ResNet50 and EfficientNetB0 are compared across both strategies.

---

## Problem Description and Motivation

Automatically classifying fine-grained plant species from images has real-world value in botanical research, ecology, and conservation. Tasks that would otherwise require expert manual identification at scale. The Oxford 102 Flowers dataset presents a challenging benchmark with 102 visually similar categories, making it a strong testbed for evaluating whether pre-trained ImageNet features transfer effectively to a new domain, and how much benefit fine-tuning the deeper layers provides over simple feature extraction.

---

## Dataset

The **Oxford 102 Flowers** dataset was used, accessed via the [PyTorch Challenge Flower Dataset](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset) on Kaggle.

| Split | Images |
|---|---|
| Train | ~6,552 |
| Validation | ~818 |
| Test | ~819 |
| **Classes** | **102** |

- **Image size:** 224 × 224 (resized)
- **Pre-existing splits** from the dataset were used directly (train / valid / test folders)
- The dataset has moderate class imbalance — some species have significantly more images than others

---

## Setup Instructions

### Requirements

- Python 3.8+
- TensorFlow 2.x
- Jupyter Notebook / Google Colab (T4 GPU recommended)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd mini-project-6
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset Structure

The notebook makes use of the dataset under the /data directory, set up like such:

```
data/
    train/
    valid/
    test/
  cat_to_name.json
```

### 4. Run the notebook

```bash
jupyter notebook notebooks/notebook.ipynb
```

Run all cells top to bottom. The notebook trains all four models (ResNet50 FE, EfficientNetB0 FE, ResNet50 FT, EfficientNetB0 FT) in sequence and displays results after each stage.

> **Google Colab:** Go to *Runtime → Change runtime type → T4 GPU* before running.

---

## Results Summary

Four models were trained and evaluated on the validation set (the test set has no labels in this dataset):

| Model | Strategy | Val Accuracy | Val Loss | Training Time | Trainable Params |
|---|---|---|---|---|---|
| ResNet50 | Feature Extraction | 91.6% | 0.3253 | 992s | ~275K |
| EfficientNetB0 | Feature Extraction | 93.3% | 0.2431 | 501s | ~177K |
| ResNet50 | Fine-Tuning | 88.1% | 0.5572 | 846s | ~14.7M |
| EfficientNetB0 | Fine-Tuning | 95.7% | 0.2079 | 422s | ~1.7M |

> **Note:** Exact accuracy figures are generated at runtime. Run the notebook to populate this table with your results.
>
> 

## Training Analysis
<img width="1589" height="495" alt="image" src="https://github.com/user-attachments/assets/e9a6a761-f512-46bf-81dc-6c5fe1620e77" />

**Training Curves (All 4 Models):**
Feature extraction models (ResNet50 and EfficientNetB0) converge quickly within the first 2–3 epochs, while ResNet50 fine-tuning trains much slower and never catches up, reflecting the cost of retraining 14.7M parameters with a very conservative learning rate.
<img width="1189" height="495" alt="image" src="https://github.com/user-attachments/assets/acabf34a-6a95-4e8f-9235-1d16f8074941" />

**Feature Extraction vs Fine-Tuning Comparison:**
EfficientNetB0 fine-tuning is the clear winner at 96% accuracy and the lowest loss (0.21), while ResNet50 fine-tuning is the worst performer despite having the most trainable parameters, suggesting it overfit or needed more epochs to recover from the low learning rate.


### Key Findings

- **Feature extraction** is fast and effective as a baseline, frozen pre-trained weights provide strong general features even for the specialised domain of flower species
- **Fine-tuning** unfreezes the deeper layers and retrains them with a much lower learning rate (1e-5 for ResNet50, 1e-4 for EfficientNetB0), allowing the model to adapt ImageNet features to flower-specific textures and shapes
- **EfficientNetB0** is more parameter-efficient than ResNet50 (5.3M vs 25.6M total parameters) while achieving competitive accuracy, making it well-suited to this dataset size
- The **Oxford 102 Flowers** dataset falls in the *"small dataset, different domain"* region of the transfer learning decision framework, enough data to benefit from fine-tuning, but not enough to train from scratch

### Bonus Experiments

- **Architecture comparison:** ResNet50 vs EfficientNetB0 evaluated head-to-head across both strategies
- **Layer unfreezing experiment:** EfficientNetB0 tested with 0, 10, 30, and 50 unfrozen layers to measure the accuracy vs. training cost tradeoff
- **Learning rate scheduling:** Constant 1e-4 vs CosineDecay vs ExponentialDecay compared over 10 epochs

---

## Repository Structure

```
mini-project-6/
├── notebooks/
│   └── notebook.ipynb        # Main notebook (all models + analysis)
├── data/
  ├── train/
  ├── test/
  └── valid/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## References

- **Dataset:** [PyTorch Challenge Flower Dataset](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset) — originally from [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) (Nilsback & Zisserman, 2008)

---

## Team Member Contributions

| Member | Contributions |
|---|---|
| **Sepehr Mansouri** | EDA, data preparation, feature extraction models, fine-tuning models, bonus experiments, comparison & analysis, report, and all other requirements |
