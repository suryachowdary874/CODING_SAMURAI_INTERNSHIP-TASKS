# Handwritten Digits Recognition

A complete end-to-end Machine Learning pipeline to recognize handwritten digits (0–9) using the **Scikit-learn Digits dataset** and a **Multi-Layer Perceptron (MLP) Classifier**.

---

## Project Overview

This project walks through the full ML workflow — from data loading and exploration to preprocessing, model training, evaluation, and prediction from real image files.

---

## Project Structure

```
digits_recognition.py   # Full pipeline in a single script
scaled_data.pkl         # Saved preprocessed data (auto-generated)
model.pkl               # Saved trained model (auto-generated)
0.png ... 6.png         # Optional: custom digit images for prediction
README.md
```

---

## Pipeline Stages

| Step | Description |
|------|-------------|
| **1. Load & Explore** | Load digits dataset, analyze shapes, class distribution & pixel range |
| **2. Preprocessing** | Flatten images, train/test split, StandardScaler normalization |
| **3. Train & Evaluate** | Train MLPClassifier, print accuracy & classification report |
| **4. Predict** | Predict digits from custom `.png` image files |
| **5. Save** | Persist trained model as `model.pkl` |

---

## Model Architecture

- **Model:** MLPClassifier (Multi-Layer Perceptron)
- **Hidden Layer:** 1 layer x 64 neurons
- **Activation:** ReLU
- **Optimizer:** Adam
- **Early Stopping:** Enabled (patience: 10 iterations)
- **Input Features:** 64 (flattened 8x8 image)
- **Output Classes:** 10 (digits 0–9)

---

## Dataset

- **Source:** `sklearn.datasets.load_digits`
- **Samples:** 1,797 images
- **Image Size:** 8x8 grayscale pixels
- **Classes:** 10 (digits 0 through 9)
- **Missing Values:** None

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/digits-recognition.git
cd digits-recognition
```

### 2. Install Dependencies
```bash
pip install scikit-learn numpy pillow
```

### 3. Run the Pipeline
```bash
python digits_recognition.py
```

---

## Sample Output

```
STEP 1: Load & Explore Data
Total samples       : 1797
Image shape         : (8, 8)
Number of classes   : 10
Pixel value range   : (0.0, 16.0)
Has missing values? : False

STEP 3: Training & Evaluation
Test Accuracy: 0.9750

Classification Report:
              precision    recall  f1-score
           0       1.00      1.00      1.00
           1       0.97      0.97      0.97
           ...
```

---

## Predict from Your Own Images

Place grayscale digit images named `0.png`, `1.png`, ... `6.png` in the project folder. The pipeline will automatically detect and predict them.

```
0.png -> Predicted: 0
3.png -> Predicted: 3
```

> Images are auto-resized to 8x8 before prediction.

---

## Tech Stack

- **Language:** Python 3.x
- **Libraries:** Scikit-learn, NumPy, Pillow
- **Model:** MLPClassifier (Neural Network)

---

## Author

**Surya Chowdary**  