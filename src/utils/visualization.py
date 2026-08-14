"""
Visualization Suite generating the 6 Standard Experiment Graphs (Section 7.2).
"""

import os
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless CLI execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10})


def plot_01_optimization_history(trials_df: pd.DataFrame, output_path: str):
    """Plot 1: Hyperparameter Optimization History Plot (F1 / F2 vs Trial)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if "trial" in trials_df.columns and "score" in trials_df.columns:
        ax.plot(trials_df["trial"], trials_df["score"], marker="o", color="#2b5c8f", alpha=0.7, label="Trial Score")
        ax.plot(trials_df["trial"], trials_df["score"].cummax(), color="#d95f02", linewidth=2.5, label="Best Score So Far")
        ax.set_title("Optuna Hyperparameter Optimization History", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Optimization Score (F1 / F2)")
        ax.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_02_model_performance_comparison(df_metrics: pd.DataFrame, output_path: str, task_name: str = "Task 1"):
    """Plot 2: Model / Approach Performance Comparison Bar Chart."""
    fig, ax = plt.subplots(figsize=(11, 6))
    model_col = "approach" if "approach" in df_metrics.columns else ("model" if "model" in df_metrics.columns else None)
    metrics_to_plot = [
        c for c in ["f1", "f2", "precision", "recall", "accuracy", "mcc",
                    "phone_exact_match", "location_exact_match", "count_exact_match"]
        if c in df_metrics.columns
    ]
    
    if model_col and len(metrics_to_plot) > 0:
        melted = df_metrics.melt(id_vars=[model_col], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
        sns.barplot(data=melted, x=model_col, y="Score", hue="Metric", ax=ax, palette="Set2")
        ax.set_title(f"Model Performance Comparison - {task_name}", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        plt.xticks(ha="right")
        ax.set_xlabel("Model / Approach")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_task2_extraction_comparison(df_metrics: pd.DataFrame, graphs_dir: str):
    """Generates dedicated Task 2 plots: (1) F1/F2/Exact Match Scores, (2) Count MAE & RMSE Errors."""
    model_col = "approach" if "approach" in df_metrics.columns else ("model" if "model" in df_metrics.columns else None)
    if not model_col:
        return
        
    os.makedirs(graphs_dir, exist_ok=True)
    
    # 1. Scores Comparison (F1, F2, Phone Match, Location Match, Count Exact Match) - Higher is better
    score_cols = [c for c in ["f1", "f2", "count_exact_match", "phone_exact_match", "location_exact_match", "map_url_exact_match", "coords_exact_match"] if c in df_metrics.columns]
    if score_cols:
        fig, ax = plt.subplots(figsize=(11, 6))
        melted = df_metrics.melt(id_vars=[model_col], value_vars=score_cols, var_name="Metric", value_name="Score")
        sns.barplot(data=melted, x=model_col, y="Score", hue="Metric", ax=ax, palette="Set2")
        ax.set_title("Task 2 Extraction: F1/F2 & Accuracy Scores (Higher is Better)", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        plt.xticks(ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Approach / Model")
        ax.set_ylabel("Score (0.0 - 1.0)")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "02_task2_extraction_scores_comparison.png"), dpi=300)
        plt.close()
        
    # 2. Error Metrics Comparison (Count MAE, Count RMSE) - Lower is better
    err_cols = [c for c in ["mean_count_mae", "mean_count_rmse"] if c in df_metrics.columns]
    if err_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        melted_err = df_metrics.melt(id_vars=[model_col], value_vars=err_cols, var_name="Error Metric", value_name="Error Value")
        sns.barplot(data=melted_err, x=model_col, y="Error Value", hue="Error Metric", ax=ax, palette="Reds")
        ax.set_title("Task 2 Extraction: Count Error Rates (MAE & RMSE - Lower is Better)", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        plt.xticks(ha="right")
        ax.set_xlabel("Approach / Model")
        ax.set_ylabel("Error Value (Count Difference)")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "02_task2_extraction_error_mae_rmse.png"), dpi=300)
        plt.close()


def plot_task4_e2e_summary(e2e_metrics: Dict[str, Any], output_path: str):
    """Plots Task 4 End-to-End Integrated Pipeline Summary Bar Chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    keys = ["overall_e2e_f1", "overall_e2e_f2", "overall_e2e_accuracy", "clinical_qwk"]
    labels = ["E2E F1-Score", "E2E F2-Score", "E2E Accuracy", "Clinical QWK"]
    values = [float(e2e_metrics.get(k, 0.0)) for k in keys]
    
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"]
    bars = ax.bar(labels, values, color=colors, width=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")
                    
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title("Task 4: End-to-End Integrated Pipeline Benchmark", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_03_f1_f2_vs_latency_tradeoff(df_tradeoff: pd.DataFrame, output_path: str):
    """Plot 3: F1/F2 Score vs p95 Latency Trade-Off Scatter Plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if "p95_latency_ms" in df_tradeoff.columns and "f1" in df_tradeoff.columns:
        scatter = ax.scatter(
            df_tradeoff["p95_latency_ms"], df_tradeoff["f1"],
            c=df_tradeoff.get("f2", df_tradeoff["f1"]), cmap="viridis", s=120, edgecolors="k", alpha=0.8
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("F2-Score")
        
        for idx, row in df_tradeoff.iterrows():
            ax.annotate(str(row["model"]), (row["p95_latency_ms"], row["f1"]),
                        textcoords="offset points", xytext=(5, 5), ha="left", fontsize=9)
            
        ax.set_xscale("log")
        ax.set_title("Accuracy (F1) vs p95 Inference Latency Trade-Off", fontsize=12, fontweight="bold")
        ax.set_xlabel("p95 Latency (ms, log-scale)")
        ax.set_ylabel("F1 Score")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_04_confusion_matrix_heatmap(cm: np.ndarray, labels: List[str], output_path: str, title: str = "Confusion Matrix"):
    """Plot 4: Confusion Matrix Heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_05_feature_importance(features: List[str], importances: List[float], output_path: str, top_n: int = 20):
    """Plot 5: Feature Importance / Top N-gram Weights."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(features) > 0:
        top_idx = np.argsort(importances)[-top_n:]
        top_feats = [features[i] for i in top_idx]
        top_scores = [importances[i] for i in top_idx]
        
        ax.barh(range(len(top_feats)), top_scores, color="#3182bd")
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats)
        ax.set_title(f"Top {top_n} Feature Importances / Weights", fontsize=12, fontweight="bold")
        ax.set_xlabel("Weight / Importance Score")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_06_cv_val_vs_test(df_gap: pd.DataFrame, output_path: str):
    """Plot 6: Cross-Dataset Performance Gap Comparison Chart (5-Fold CV vs Held-out Test)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    test_score_col = "test_score" if "test_score" in df_gap.columns else ("luna_score" if "luna_score" in df_gap.columns else None)
    if "model" in df_gap.columns and "cv_score" in df_gap.columns and test_score_col:
        x = np.arange(len(df_gap))
        width = 0.35
        
        ax.bar(x - width/2, df_gap["cv_score"], width, label="5-Fold CV", color="#4292c6")
        ax.bar(x + width/2, df_gap[test_score_col], width, label="Held-out Test", color="#ef3b2c")
        
        ax.set_xticks(x)
        ax.set_xticklabels(df_gap["model"], rotation=30, ha="right")
        ax.set_ylabel("F1 / Score")
        ax.set_title("Cross-Dataset Performance Gap (5-Fold CV vs Held-out Test)", fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


# Legacy alias for backwards compatibility
plot_06_gemini_val_vs_luna_test = plot_06_cv_val_vs_test
