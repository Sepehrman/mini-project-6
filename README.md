# Mini Project 6

A Convolutional Neural Network that classifies satellite images into 4 categories: **Cloudy, Desert, Green Area, and Water**.

---

## Problem Description and Motivation

Being able to automatically identify different types of land cover from satellite images is useful for a wide range of environmental monitoring tasks. For example, tracking desertification over time, monitoring crop health and farm outputs, detecting habitat loss for wildlife conservation, and observing changes in sea ice or glacier coverage all depend on reliably telling different terrain types apart across large areas. Doing this by hand from satellite data is slow and expensive. A trained CNN can classify thousands of images in seconds, making large-scale monitoring far more practical.

This project trains and compares three CNN architectures on a 4-class satellite image dataset to explore how well simple convolutional networks handle this task, and what effect data augmentation and architectural changes have on performance.

---

## Dataset Description

The dataset contains **5,631 satellite images** split across 4 classes, sourced from Kaggle.

| Class | Images |
|---|---|
| Cloudy | 1,500 |
| Desert | 1,131 |
| Green Area | 1,500 |
| Water | 1,500 |
| **Total** | **5,631** |

- **Image size:** 250 x 250 pixels
- **Train / Validation split:** 80% / 20%
- **Source:** [Kaggle - Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)

### Sample Images Per Class

<img width="132" height="634" alt="image" src="https://github.com/user-attachments/assets/dde151d4-7853-425e-bd26-e8c8f4369ec9" />


### Class Distribution
<img width="870" height="864" alt="image" src="https://github.com/user-attachments/assets/dfbc5e99-a21e-4052-b3da-b7a412795358" />


---

## Setup and Running Instructions

### Requirements

- Python 3.8+
- TensorFlow 2.x
- Jupyter Notebook

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/RealGoldenGeneral/mini-project-5.git
cd mini-project-5
```

**2. Install required libraries**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification) and place it in a `data/` folder at the root of the project:

```
data/
  cloudy/
  desert/
  green_area/
  water/
```

**4. Run the notebook**
```bash
cd notebooks
jupyter notebook cnn_model.ipynb
```

Run all cells from top to bottom. The notebook trains the Baseline CNN, the Augmented CNN, and the Bonus GAP CNN in sequence and displays results after each.

---

## Results Summary

Three models were trained and compared:

| Model | Train Accuracy | Val Accuracy | F1 Score | Train-Val Gap |
|---|---|---|---|---|
| Baseline CNN | 0.9487 (94.9%) | 0.9174 (91.7%) | 0.9171 | 0.0313 (3.1%) |
| Augmented CNN | 0.7738 (77.4%) | 0.7131 (71.3%) | 0.6203 | 0.0607 (6.1%) |
| GAP CNN (Bonus) | 0.9400 (94.0%) | 0.9400 (94.0%) | 0.9400 | 0.0020 (0.2%) |

### Key Findings

- The **Baseline CNN** achieved 91.7% validation accuracy with a tight 3.1% train-val gap, showing solid generalisation with no regularisation at all.
- The **Augmented CNN** underperformed despite added BatchNorm and Dropout. The combination of aggressive contrast augmentation with BatchNormalization caused training instability, with validation loss spiking above 1,600 at epoch 16. It also completely failed on `green_area`, misclassifying every single one as `water`.
- The **GAP CNN** was the best overall with 94% validation accuracy and a near-zero 0.2% gap. Swapping `Flatten` for `GlobalAveragePooling2D` reduced the parameter count from ~13M to ~0.5M, greatly cutting overfitting.

---

## Sample Predictions

### Baseline CNN
*Green = correct, Red = wrong*

<img width="721" height="244" alt="image" src="https://github.com/user-attachments/assets/5ca065a6-2715-4740-bac3-8ef9c59cadc9" />


The baseline model predicts confidently and correctly across all four classes in this batch, with most confidence scores at 0.99-1.00.

### Augmented CNN
*Green = correct, Red = wrong*

<img width="766" height="268" alt="image" src="https://github.com/user-attachments/assets/dd11da33-fa0b-4aeb-a62a-09c628a70fee" />


The augmented model misclassifies `green_area` as `water` (shown in red). This matches the confusion matrix result where all 318 validation `green_area` images were predicted as water.

### Misclassified Examples (Augmented CNN)

<img width="1384" height="794" alt="image" src="https://github.com/user-attachments/assets/09323f1c-ec0f-4f0d-81e8-e37ee5c0c884" />

All 12 shown are `green_area` images wrongly predicted as `water`. Many look visually very similar to water from satellite altitude, especially after contrast augmentation is applied during training, which removes the subtle colour differences the model needs to tell them apart.

---

### Other Planned Improvements

- **Learning rate scheduling** — adding `ReduceLROnPlateau` would likely have prevented the augmented model's loss spike at epoch 16 by automatically reducing the learning rate when validation loss stops improving
- **Milder augmentation** — reducing or removing `RandomContrast` should fix the green_area vs water confusion, since that augmentation destroys the colour difference the model needs to separate those two classes
- **Transfer learning** — using a pretrained backbone like EfficientNetB0 or MobileNetV2 would give a much stronger starting point than training from scratch and would likely outperform all three current architectures
- **Proper test set** — splitting the data into train / validation / test (e.g. 70/15/15) would give a more honest measure of how well the models actually generalise to unseen data

---

## Team Member Contributions

| Member | Contributions |
|---|---|
| **Sepehr Mansouri** | EDA, Modelling, Comparison & Analysis, Report and all other requirements |
