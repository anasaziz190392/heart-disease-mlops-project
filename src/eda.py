"""
eda.py
------
Exploratory Data Analysis for the cleaned Heart Disease dataset.
Generates professional plots (histograms, correlation heatmap, class
balance, boxplots by target) and saves them to reports/figures/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "processed", "heart_clean.csv")
FIG_DIR = os.path.join(HERE, "..", "reports", "figures")

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_class_balance(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = df["target"].value_counts().sort_index()
    labels = ["No Disease (0)", "Disease (1)"]
    colors = ["#4C72B0", "#C44E52"]
    bars = ax.bar(labels, counts.values, color=colors)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", fontweight="bold")
    ax.set_title("Class Balance: Heart Disease Target", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Patients")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "class_balance.png"), dpi=150)
    plt.close(fig)


def plot_histograms(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#4C72B0")
        axes[i].set_title(f"Distribution of {col}", fontweight="bold")
    # last panel: age split by target
    sns.histplot(data=df, x="age", hue="target", kde=True, ax=axes[5], palette=["#4C72B0", "#C44E52"])
    axes[5].set_title("Age Distribution by Target", fontweight="bold")
    fig.suptitle("Feature Distributions (Numeric Variables)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "histograms.png"), dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap of All Features", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(18, 4.5))
    for i, col in enumerate(NUMERIC_COLS):
        sns.boxplot(data=df, x="target", y=col, ax=axes[i], palette=["#4C72B0", "#C44E52"])
        axes[i].set_xticklabels(["No Disease", "Disease"])
        axes[i].set_title(col, fontweight="bold")
    fig.suptitle("Numeric Feature Spread by Target Class", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boxplots_by_target.png"), dpi=150)
    plt.close(fig)


def plot_categorical_vs_target(df: pd.DataFrame):
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col], df["target"], normalize="index")
        ct.plot(kind="bar", stacked=True, ax=axes[i], color=["#4C72B0", "#C44E52"], legend=False)
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["No Disease", "Disease"], loc="upper right")
    fig.suptitle("Categorical Features vs Target (Proportion)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "categorical_vs_target.png"), dpi=150)
    plt.close(fig)


def summary_stats(df: pd.DataFrame):
    desc = df.describe().T
    desc.to_csv(os.path.join(FIG_DIR, "..", "summary_statistics.csv"))
    print("[eda] Summary statistics written to reports/summary_statistics.csv")
    print(df.isna().sum().rename("missing_values"))


def run():
    ensure_dir()
    df = pd.read_csv(DATA_PATH)
    plot_class_balance(df)
    plot_histograms(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    plot_categorical_vs_target(df)
    summary_stats(df)
    print(f"[eda] Figures written to {FIG_DIR}")


if __name__ == "__main__":
    run()
