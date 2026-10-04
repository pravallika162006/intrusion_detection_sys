"""
Feature Importance Visualizer for Phase 2.
Generates bar charts for the ranked features and the selected top 19 features
matching Sydney M. Kasongo & Yanxia Sun (2020) Table 3.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from backend.config import RESULTS_DIR, PHASE2_RESULTS_DIR
from backend.utils.logger import setup_logger

logger = setup_logger("Visualizer")

PLOTS_DIR = PHASE2_RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_feature_importance_plots(df_ranking: pd.DataFrame, top_k: int = 19) -> None:
    """
    Generates and saves feature importance visualizations:
    1. Full 42-feature ranked bar chart (highlighting top K)
    2. Selected Top 19 features bar chart
    """
    logger.info("Generating feature importance visualizations...")
    sns.set_theme(style="whitegrid")

    score_col = "Paper_Importance_Score" if "Paper_Importance_Score" in df_ranking.columns else (
        "Importance_Score" if "Importance_Score" in df_ranking.columns else "Our_Measured_Score"
    )

    # Chart 1: All Features Ranked
    fig, ax = plt.subplots(figsize=(12, 14))
    colors = ["#0284c7" if r <= top_k else "#94a3b8" for r in df_ranking["Rank"]]
    
    sns.barplot(
        data=df_ranking,
        x=score_col,
        y="Feature",
        palette=colors,
        ax=ax,
    )
    
    ax.set_title(f"UNSW-NB15 XGBoost Feature Importance (Top {top_k} Features Highlighted)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Feature Importance Score (Kasongo & Sun, 2020 Table 3)", fontsize=12, labelpad=10)
    ax.set_ylabel("UNSW-NB15 Features", fontsize=12, labelpad=10)
    
    for index, row in df_ranking.iterrows():
        val = row[score_col]
        if val > 0:
            ax.text(
                val + 0.001,
                index,
                f"#{row['Rank']} ({val:.6f})",
                va="center",
                fontsize=8,
                color="#0f172a",
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
        x=score_col,
        y="Feature",
        palette="crest",
        ax=ax,
    )
    
    ax.set_title(f"Top {top_k} XGBoost Selected Features (Journal of Big Data 2020 Table 3)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Importance Score", fontsize=11, labelpad=10)
    ax.set_ylabel("Selected Features", fontsize=11, labelpad=10)

    for index, row in df_top19.iterrows():
        val = row[score_col]
        ax.text(
            val + 0.001,
            index,
            f"{val:.6f}",
            va="center",
            fontsize=9,
            color="#0f172a",
        )

    plt.tight_layout()
    plot_19_path = PLOTS_DIR / "top19_features.png"
    fig.savefig(plot_19_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved 42-feature plot to: {plot_42_path}")
    logger.info(f"Saved Top-19 feature plot to: {plot_19_path}")
