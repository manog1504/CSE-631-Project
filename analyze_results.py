"""Analyze experiment results and generate figures"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from evaluation.metrics import evaluate_all_transcripts
from config import TRANSCRIPT_DIR, RESULTS_DIR, FIGURES_DIR


def main():
    """Run analysis and generate all outputs"""
    # Create output directories
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    Path(FIGURES_DIR).mkdir(exist_ok=True)

    # Load topics
    with open("topics.json", "r") as f:
        topics = json.load(f)

    print("Analyzing experiment results...")
    print("-" * 60)

    # Compute all metrics
    results_df = evaluate_all_transcripts(TRANSCRIPT_DIR, topics)

    if results_df.empty:
        print("Error: No transcripts found. Run run_experiments.py first.")
        return

    # Save metrics CSV
    csv_path = Path(RESULTS_DIR) / "metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Metrics saved to {csv_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    summary = results_df.groupby("protocol")[["factual_accuracy", "position_shift", "agreement"]].agg([
        "mean", "std", "min", "max"
    ])
    print(summary)

    # Generate Figure 1: Factual Accuracy per Protocol
    fig, ax = plt.subplots(figsize=(8, 6))
    protocols = ["A", "B", "C"]
    accuracy_means = [results_df[results_df["protocol"] == p]["factual_accuracy"].mean() for p in protocols]
    accuracy_stds = [results_df[results_df["protocol"] == p]["factual_accuracy"].std() for p in protocols]

    bars = ax.bar(protocols, accuracy_means, yerr=accuracy_stds, capsize=10, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Factual Accuracy", fontsize=12)
    ax.set_xlabel("Protocol", fontsize=12)
    ax.set_title("Factual Accuracy by Protocol", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1.0])
    ax.grid(axis="y", alpha=0.3)

    for i, (mean, std) in enumerate(zip(accuracy_means, accuracy_stds)):
        ax.text(i, mean + std + 0.05, f"{mean:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "01_factual_accuracy.png", dpi=300, bbox_inches="tight")
    print(f"✓ Figure saved: 01_factual_accuracy.png")
    plt.close()

    # Generate Figure 2: Position Shift Over Rounds (B and C only)
    fig, ax = plt.subplots(figsize=(10, 6))

    for proto in ["B", "C"]:
        proto_data = results_df[results_df["protocol"] == proto]["position_shift"]
        label = f"Protocol {proto} (Sequential)" if proto == "B" else f"Protocol {proto} (Judge-Mediated)"
        # Since we only have mean shifts, plot them as horizontal lines
        ax.axhline(y=proto_data.mean(), label=label, linewidth=2, linestyle="-" if proto == "B" else "--")

    ax.set_ylabel("Mean Position Shift (Cosine Distance)", fontsize=12)
    ax.set_xlabel("Protocol", fontsize=12)
    ax.set_title("Position Shift Convergence (Protocols B & C)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "02_position_shift.png", dpi=300, bbox_inches="tight")
    print(f"✓ Figure saved: 02_position_shift.png")
    plt.close()

    # Generate Figure 3: Inter-Agent Agreement at Termination
    fig, ax = plt.subplots(figsize=(8, 6))
    protocols = ["A", "B", "C"]
    agreement_means = [results_df[results_df["protocol"] == p]["agreement"].mean() for p in protocols]
    agreement_stds = [results_df[results_df["protocol"] == p]["agreement"].std() for p in protocols]

    boxes = ax.boxplot(
        [results_df[results_df["protocol"] == p]["agreement"].values for p in protocols],
        labels=protocols,
        patch_artist=True,
        widths=0.6
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Agreement (Cosine Similarity)", fontsize=12)
    ax.set_xlabel("Protocol", fontsize=12)
    ax.set_title("Inter-Agent Agreement at Termination", fontsize=14, fontweight="bold")
    ax.set_ylim([-0.1, 1.1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "03_interagent_agreement.png", dpi=300, bbox_inches="tight")
    print(f"✓ Figure saved: 03_interagent_agreement.png")
    plt.close()

    # Generate Figure 4: Comparison Matrix (all metrics)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["factual_accuracy", "position_shift", "agreement"]
    metric_labels = ["Factual Accuracy", "Position Shift", "Agreement"]

    for ax, metric, label in zip(axes, metrics, metric_labels):
        data = [results_df[results_df["protocol"] == p][metric].values for p in protocols]
        bp = ax.boxplot(data, labels=protocols, patch_artist=True, widths=0.6)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "04_all_metrics_boxplot.png", dpi=300, bbox_inches="tight")
    print(f"✓ Figure saved: 04_all_metrics_boxplot.png")
    plt.close()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
