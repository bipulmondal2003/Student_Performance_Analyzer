"""
visualize.py
Generates all data visualisation figures for the Streamlit app.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── Theme constants ───────────────────────────────────────────────────────────
BG_MAIN   = "#0e1117"
BG_PLOT   = "#1a1a2e"
PASS_COLOR = "#2ecc71"
FAIL_COLOR = "#e74c3c"
LR_COLOR   = "#3498db"
DT_COLOR   = "#9b59b6"
TEXT_COLOR = "white"


def _apply_dark_theme(fig, ax):
    """Apply consistent dark theme to a figure/axes pair."""
    fig.patch.set_facecolor(BG_MAIN)
    ax.set_facecolor(BG_PLOT)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def scatter_study_vs_result(df: pd.DataFrame):
    """Scatter: Study Hours vs Result (PASS/FAIL)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    _apply_dark_theme(fig, ax)

    for label, color, marker in [(0, FAIL_COLOR, "x"), (1, PASS_COLOR, "o")]:
        subset = df[df["result"] == label]
        ax.scatter(
            subset["study_hours"], subset["previous_marks"],
            c=color, marker=marker, alpha=0.75, s=60,
            label="PASS" if label == 1 else "FAIL",
            edgecolors="white" if label == 1 else "none", linewidths=0.5
        )

    ax.set_xlabel("Study Hours / Day")
    ax.set_ylabel("Previous Marks")
    ax.set_title("Study Hours vs Previous Marks  |  PASS / FAIL")
    ax.legend(facecolor="#222", labelcolor=TEXT_COLOR, framealpha=0.8)
    plt.tight_layout()
    return fig


def scatter_attendance_vs_result(df: pd.DataFrame):
    """Scatter: Attendance % vs Previous Marks, coloured by result."""
    fig, ax = plt.subplots(figsize=(7, 4))
    _apply_dark_theme(fig, ax)

    for label, color, marker in [(0, FAIL_COLOR, "x"), (1, PASS_COLOR, "o")]:
        subset = df[df["result"] == label]
        ax.scatter(
            subset["attendance"], subset["previous_marks"],
            c=color, marker=marker, alpha=0.75, s=60,
            label="PASS" if label == 1 else "FAIL",
            edgecolors="white" if label == 1 else "none", linewidths=0.5
        )

    ax.axvline(70, color="#e67e22", linestyle="--", linewidth=1.2, label="70% threshold")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Previous Marks")
    ax.set_title("Attendance vs Previous Marks  |  PASS / FAIL")
    ax.legend(facecolor="#222", labelcolor=TEXT_COLOR, framealpha=0.8)
    plt.tight_layout()
    return fig


def bar_model_accuracy(results: dict):
    """Bar chart comparing model accuracies."""
    names, accs, colors = [], [], [LR_COLOR, DT_COLOR]

    for k, v in results.items():
        if k in ("best_model_name", "best_model"):
            continue
        names.append(k)
        accs.append(v["accuracy"])

    fig, ax = plt.subplots(figsize=(6, 4))
    _apply_dark_theme(fig, ax)

    bars = ax.bar(names, accs, color=colors[:len(names)],
                  edgecolor="white", linewidth=0.5, width=0.45)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.008,
            f"{acc*100:.1f}%",
            ha="center", va="bottom", color=TEXT_COLOR, fontweight="bold", fontsize=11
        )

    # Highlight best
    best_name = results.get("best_model_name", "")
    for bar, name in zip(bars, names):
        if name == best_name:
            bar.set_edgecolor("#f1c40f")
            bar.set_linewidth(2.5)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")

    # Legend for best model marker
    gold_patch = mpatches.Patch(edgecolor="#f1c40f", facecolor="none",
                                linewidth=2, label=f"Best: {best_name}")
    ax.legend(handles=[gold_patch], facecolor="#222", labelcolor=TEXT_COLOR, framealpha=0.8)
    plt.tight_layout()
    return fig


def histogram_features(df: pd.DataFrame):
    """Distribution histograms for each feature, split by result."""
    features = ["study_hours", "attendance", "previous_marks"]
    labels_map = {0: ("FAIL", FAIL_COLOR), 1: ("PASS", PASS_COLOR)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(BG_MAIN)

    for ax, feat in zip(axes, features):
        _apply_dark_theme(fig, ax)
        for result_val, (lbl, clr) in labels_map.items():
            subset = df[df["result"] == result_val][feat]
            ax.hist(subset, bins=12, color=clr, alpha=0.65, label=lbl, edgecolor="none")

        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel(feat.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.legend(facecolor="#222", labelcolor=TEXT_COLOR, fontsize=8, framealpha=0.8)

    fig.suptitle("Feature Distributions by Outcome", color=TEXT_COLOR, fontsize=13, y=1.02)
    plt.tight_layout()
    return fig
