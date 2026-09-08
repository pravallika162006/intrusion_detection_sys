"""
Feature Importance Visualizer for Phase 2.
Generates bar charts for all 42 ranked features and the selected top 19 features.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from backend.config import RESULTS_DIR
from backend.utils.logger import setup_logger

logger = setup_logger("Visualizer")

PLOTS_DIR = RESULTS_DIR / "phase2" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_feature_importance_plots(df_ranking: pd.DataFrame, top_k: int = 19) -> None:
    """
    Generates and saves feature importance visualizations:
    1. Full 42-feature ranked bar chart (highlighting top K)
    2. Selected Top 19 features bar chart
    """
    logger.info("Generating feature importance visualizations...")
    sns.set_theme(style="whitegrid")

    # Chart 1: All 42 Features Ranked
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Assign colors: Top K highlighted in vibrant navy/teal, remainder in muted gray
    colors = ["#1f77b4" if r <= top_k else "#aec7e8" for r in df_ranking["Rank"]]
    
    sns.barplot(
        data=df_ranking,
        x="Importance_Score",
        y="Feature",
        palette=colors,
        ax=ax,
    )
    
    ax.set_title(f"UNSW-NB15 XGBoost Feature Importance (All 42 Features, Top {top_k} Highlighted)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Normalized Feature Importance Score", fontsize=12, labelpad=10)
    ax.set_ylabel("Original Input Features", fontsize=12, labelpad=10)
    
    # Add rank labels inside/beside bars
    for index, row in df_ranking.iterrows():
        ax.text(
            row["Importance_Score"] + 0.001,
            index,
            f"#{row['Rank']} ({row['Importance_Score']:.4f})",
            va="center",
            fontsize=9,
            color="#333333",
        )

    plt.tight_layout()
    plot_42_path = PLOTS_DIR / "xgboost_feature_importance_42.png"
    fig.savefig(plot_42_path, dpi=300)
    plt.close(fig)

    # Chart 2: Selected Top 19 Features Only
    df_top19 = df_ranking.head(top_k).copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.barplot(
        data=df_top19,
        x="Importance_Score",
        y="Feature",
        palette="viridis",
        ax=ax,
    )
    
    ax.set_title(f"Top {top_k} XGBoost-Selected Features for Phase 2 Reduced Pipeline", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Normalized Feature Importance Score", fontsize=11, labelpad=10)
    ax.set_ylabel("Selected Features", fontsize=11, labelpad=10)

    for index, row in df_top19.iterrows():
        ax.text(
            row["Importance_Score"] + 0.001,
            index,
            f"{row['Importance_Score']:.4f}",
            va="center",
            fontsize=9,
            color="#000000",
        )

    plt.tight_layout()
    plot_19_path = PLOTS_DIR / "top19_features.png"
    fig.savefig(plot_19_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved 42-feature plot to: {plot_42_path}")
    logger.info(f"Saved Top-19 feature plot to: {plot_19_path}")
