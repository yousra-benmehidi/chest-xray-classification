# HealthLab — Chest X-Ray Disease Classifier

A convolutional neural network (CNN) built with TensorFlow/Keras to classify chest X-rays into three categories: **Normal**, **Pneumonia**, and **Lung Opacity**. Trained and evaluated on the [Normal-Pneumo-LungOp dataset](https://www.kaggle.com/datasets) from Kaggle.

---

## Overview

| Item | Detail |
|---|---|
| Task | Multi-class image classification (3 classes) |
| Input | RGB chest X-ray images, resized to 224×224 |
| Architecture | Custom CNN with Batch Normalization + L2 regularization |
| Loss | Focal Loss (γ=2.0, α=0.25) |
| Optimizer | Adam (lr=1e-4 with warm-up scheduler) |
| Final metric | Test accuracy reported after evaluation |

---

## Dataset

The dataset contains chest X-ray images organised into three subdirectories:

```
Normal Pneumo LungOp/
├── Normal/         (~10,000 images)
├── Pneumonia/      (~4,000 images)
└── Lung Opacity/   (~6,000 images)
```

The class imbalance is addressed via **class weights** (`Normal: 1.0`, `Pneumonia: 2.5`, `Lung Opacity: 1.7`) and **focal loss**, which down-weights easy examples and focuses training on hard, misclassified ones.

---

## Project Structure

```
healthlab.ipynb          # Main notebook (data → model → evaluation)
my_model.keras           # Saved model after training
```

---

## Pipeline

### 1. Data Loading & Splitting
- Images are loaded via `tf.keras.utils.image_dataset_from_directory` with labels inferred from folder names.
- Split strategy: **80/20 train-test**, then **80/20 train-val** on the training portion.
- A class distribution bar chart is generated to visualise imbalance.

### 2. Preprocessing & Augmentation
- **Normalisation**: pixel values scaled to [0, 1].
- **Augmentation** (train only): random horizontal flip, brightness jitter (±4%), and contrast jitter (±5%) — kept mild to avoid distribution shift.
- Validation and test sets receive normalisation only.

### 3. Model Architecture

```
Input (224×224×3)
→ Conv2D(32) → BN → ReLU → MaxPool → Conv2D(32, L2)
→ Conv2D(64) → BN → ReLU → MaxPool → Conv2D(32, L2)
→ Conv2D(128) → BN → ReLU → MaxPool → Conv2D(32, L2)
→ Flatten → Dense(128) → BN → ReLU → Dropout(0.5)
→ Dense(3, softmax)
```

Batch Normalization after each conv block stabilises training. L2 regularisation on intermediate conv layers reduces overfitting.

### 4. Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 20 (with early stopping, patience=4) |
| Batch size | 32 |
| Initial learning rate | 1e-4 |
| LR warm-up | epochs 0–3: 1e-5 → 7e-5 |
| LR decay | epochs 6–12: 1e-5 → 1e-6 |
| Early stopping | monitors `val_loss`, restores best weights |
| ReduceLROnPlateau | factor=0.5, patience=2, min_lr=1e-6 |

### 5. Evaluation

After training, the notebook produces:
- **Training curves**: accuracy and loss over epochs, with best-epoch marker and test-set reference lines.
- **Sample predictions**: 6 random test images with true label, predicted label, and confidence score.
- **Confusion matrix**: heatmap of true vs. predicted classes across the full test set.
- **Classification report**: per-class precision, recall, and F1-score.

---

## Requirements

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
```

The notebook is designed for the **Kaggle** environment (paths reference `/kaggle/input/`). To run locally, update `DATA_PATH` to your local dataset directory.

---

## Usage

1. Upload the dataset to Kaggle (or adjust `DATA_PATH` locally).
2. Run all cells sequentially in `healthlab.ipynb`.
3. The trained model is saved as `my_model.keras` for later inference.

To load and use the saved model:

```python
import tensorflow as tf

model = tf.keras.models.load_model("my_model.keras")
# model.predict(your_image_batch)
```

---

## Notes

- The dataset is imbalanced; do not rely on raw accuracy alone — check per-class F1 from the classification report.
- Augmentation parameters are intentionally conservative to preserve diagnostic features in medical images.
- Focal loss is preferred over cross-entropy here because it naturally handles class imbalance by penalising confident wrong predictions more heavily.
