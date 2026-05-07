"""
evaluate.py
Model evaluation: confusion matrix, cross-validation, feature importance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score


# ── Colour palette (consistent across plots) ─────────────────────────────────
PASS_COLOR = "#2ecc71"
FAIL_COLOR = "#e74c3c"
LR_COLOR   = "#3498db"
DT_COLOR   = "#9b59b6"


def get_confusion_matrix_fig(model, X_test, y_test, model_name: str):
    """Return a matplotlib figure of the confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAIL", "PASS"])
    disp.plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )
    ax.set_title(f"Confusion Matrix — {model_name}", color="white", pad=12)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")
    for text in disp.text_.ravel():
        text.set_color("black")

    plt.tight_layout()
    return fig


def get_cross_val_fig(models_dict: dict, X, y):
    """
    Return a matplotlib figure showing cross-validation scores (cv=5)
    for each model as a box-plot style bar with error bars.
    """
    names, means, stds = [], [], []

    for name, info in models_dict.items():
        if name in ("best_model_name", "best_model"):
            continue
        cv_scores = cross_val_score(info["model"], X, y, cv=5, scoring="accuracy")
        names.append(name)
        means.append(cv_scores.mean())
        stds.append(cv_scores.std())

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1a2e")

    colors = [LR_COLOR, DT_COLOR]
    bars = ax.bar(names, means, yerr=stds, capsize=8,
                  color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Annotate bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.01,
            f"{mean:.3f} ± {std:.3f}",
            ha="center", va="bottom", color="white", fontsize=9
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy (CV=5)", color="white")
    ax.set_title("Cross-Validation Scores", color="white", pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0e1117")
    plt.tight_layout()
    return fig


def get_feature_importance_fig(dt_model, feature_names: list):
    """Return a matplotlib figure of Decision Tree feature importances."""
    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1a2e")

    bar_colors = [DT_COLOR, "#e67e22", "#1abc9c"]
    ax.barh(sorted_names[::-1], sorted_vals[::-1],
            color=bar_colors[:len(sorted_names)], edgecolor="white", linewidth=0.5)

    for i, (val, name) in enumerate(zip(sorted_vals[::-1], sorted_names[::-1])):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center", color="white", fontsize=9)

    ax.set_xlabel("Importance Score", color="white")
    ax.set_title("Feature Importance — Decision Tree", color="white", pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    plt.tight_layout()
    return fig


def get_cv_scores(model, X, y, cv: int = 5) -> np.ndarray:
    """Return raw cross-validation scores."""
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy")
