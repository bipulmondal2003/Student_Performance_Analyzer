"""
data_processing.py
Handles loading and preprocessing of the student dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load dataset from CSV file."""
    if filepath is None:
        # Resolve path relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "data.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Split features and target, scale features, return train/test splits.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    features = ["study_hours", "attendance", "previous_marks"]
    target = "result"

    X = df[features].values
    y = df[target].values

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def get_feature_names() -> list:
    """Return feature column names."""
    return ["study_hours", "attendance", "previous_marks"]
