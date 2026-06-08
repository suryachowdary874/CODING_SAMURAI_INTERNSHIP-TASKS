# ============================================================
#         Handwritten Digits Recognition - Full Pipeline
#         Dataset: Scikit-learn Digits (8x8 grayscale images)
# ============================================================

import numpy as np
import pickle
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image


# ─────────────────────────────────────────────
# STEP 1: Load & Explore Data
# ─────────────────────────────────────────────

def load_data():
    """Load the digits dataset into arrays."""
    digits = load_digits()
    X = digits.images       # Shape: (n_samples, 8, 8)
    y = digits.target       # Shape: (n_samples,)
    return X, y


def explore_data(X, y):
    """
    Analyze the dataset: shapes, class distribution, and value range.

    Args:
        X (np.ndarray): Images array
        y (np.ndarray): Labels array

    Returns:
        dict: Dictionary containing dataset information
    """
    info = {}

    info['num_samples'] = X.shape[0]
    info['image_shape'] = X.shape[1:]          # (8, 8)

    classes = np.unique(y)
    info['num_classes'] = len(classes)

    class_counts = {}
    for cls in classes:
        class_counts[cls] = int(np.sum(y == cls))
    info['class_distribution'] = class_counts

    info['pixel_range'] = (X.min(), X.max())
    info['has_missing_values'] = bool(np.isnan(X).any() or np.isnan(y).any())

    return info


# ─────────────────────────────────────────────
# STEP 2: Preprocessing
# ─────────────────────────────────────────────

def flatten_images(X):
    """Flatten 2D (8x8) images into 1D feature vectors (n_samples, 64)."""
    return X.reshape(X.shape[0], -1)


def split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.

    Returns:
        tuple: Scaled X_train, scaled X_test, fitted scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def save_scaled_data(X_train, X_test, y_train, y_test, scaler, path="scaled_data.pkl"):
    """Save scaled data and scaler to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler
        }, f)
    print(f"Scaled data saved to '{path}'")


def load_scaled_data(file_path="scaled_data.pkl"):
    """Load preprocessed scaled data and scaler from a pickle file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"], data["scaler"]


# ─────────────────────────────────────────────
# STEP 3: Model Building, Training & Evaluation
# ─────────────────────────────────────────────

def build_model():
    """Create and return an MLPClassifier with defined hyperparameters."""
    model = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation='relu',
        solver='adam',
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    )
    return model


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Train the model and evaluate on test data."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model


def predict_digits_from_images(model, scaler, image_files):
    """
    Predict digits from a list of grayscale image files.

    Args:
        model: Trained MLPClassifier
        scaler: Fitted StandardScaler
        image_files (list): List of image file paths

    Returns:
        dict: Mapping from filename to predicted digit
    """
    predictions = {}
    for img_file in image_files:
        img = Image.open(img_file).resize((8, 8))
        img_array = np.array(img)
        img_scaled = scaler.transform(img_array.reshape(1, -1))
        predictions[img_file] = model.predict(img_scaled)[0]
    return predictions


def save_model(model, path="model.pkl"):
    """Save trained model to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to '{path}'")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: Load & Explore ──────────────────
    print("=" * 45)
    print("         STEP 1: Load & Explore Data")
    print("=" * 45)

    X, y = load_data()
    info = explore_data(X, y)

    print(f"Total samples       : {info['num_samples']}")
    print(f"Image shape         : {info['image_shape']}")
    print(f"Number of classes   : {info['num_classes']}")
    print(f"Pixel value range   : {info['pixel_range']}")
    print(f"Has missing values? : {info['has_missing_values']}")
    print("\nClass distribution:")
    for cls, count in info['class_distribution'].items():
        print(f"  Class {cls}: {count} samples")

    # ── Step 2: Preprocess ──────────────────────
    print("\n" + "=" * 45)
    print("         STEP 2: Preprocessing")
    print("=" * 45)

    X_flat = flatten_images(X)
    X_train, X_test, y_train, y_test = split_data(X_flat, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    save_scaled_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler)

    print(f"Training data shape : {X_train_scaled.shape}")
    print(f"Testing data shape  : {X_test_scaled.shape}")

    # ── Step 3: Train & Evaluate ────────────────
    print("\n" + "=" * 45)
    print("         STEP 3: Training & Evaluation")
    print("=" * 45)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_scaled_data()
    model = build_model()
    trained_model = train_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # ── Step 4: Predict from Images ─────────────
    image_files = [f"{i}.png" for i in range(7) if os.path.exists(f"{i}.png")]
    if image_files:
        print("\n" + "=" * 45)
        print("         STEP 4: Predict from Images")
        print("=" * 45)
        predictions = predict_digits_from_images(trained_model, scaler, image_files)
        for img_file, pred in predictions.items():
            print(f"  {img_file} → Predicted: {pred}")

    # ── Step 5: Save Model ──────────────────────
    save_model(trained_model)