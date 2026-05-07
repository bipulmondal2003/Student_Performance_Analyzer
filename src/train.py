"""
train.py
Handles training of Logistic Regression and Decision Tree models.
"""

import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train and return a Logistic Regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """Train and return a Decision Tree model."""
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Train both models, evaluate on test set, and return a results dict.
    Returns:
        {
            "Logistic Regression": {"model": ..., "accuracy": ...},
            "Decision Tree":       {"model": ..., "accuracy": ...},
            "best_model_name": ...,
            "best_model": ...,
        }
    """
    results = {}

    # --- Logistic Regression ---
    lr = train_logistic_regression(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    results["Logistic Regression"] = {"model": lr, "accuracy": round(lr_acc, 4)}

    # --- Decision Tree ---
    dt = train_decision_tree(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    results["Decision Tree"] = {"model": dt, "accuracy": round(dt_acc, 4)}

    # --- Determine best model ---
    best_name = max(
        ["Logistic Regression", "Decision Tree"],
        key=lambda k: results[k]["accuracy"]
    )
    results["best_model_name"] = best_name
    results["best_model"] = results[best_name]["model"]

    return results


def save_model(model, scaler, filepath: str = None):
    """Persist the best model + scaler to disk."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "models", "model.pkl")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)


def load_model(filepath: str = None):
    """Load a persisted model + scaler from disk."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "models", "model.pkl")

    if not os.path.exists(filepath):
        return None, None

    with open(filepath, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"]
