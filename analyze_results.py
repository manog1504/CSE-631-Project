"""Analyze experiment results and generate figures"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from evaluation.metrics import evaluate_all_transcripts, EvaluationEngine
from config import TRANSCRIPT_DIR, RESULTS_DIR, FIGURES_DIR


def build_per_round_shift_df(transcript_dir, topics_data):
    """
    Extract per-round position shift (cosine distance between consecutive rounds)
    for Protocols B and C. Returns a long-form DataFrame.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    rows = []

    for proto in ["B", "C"]:
        for topic_id in range(1, 11):
            filename = Path(transcript_dir) / f"protocol_{proto.lower()}_topic_{topic_id}.json"
            if not filename.exists():
                continue
            with open(filename) as f:
                transcript = json.load(f)

            rounds = transcript.get("rounds", [])
            if len(rounds) < 2:
                continue

            for agent_key in ["agent1", "agent2"]:
                args = [
                    r[agent_key]["argument"] for r in rounds
                    if agent_key in r and r[agent_key].get("argument")
                    and not r[agent_key]["argument"].startswith("[Failed")
                ]
                if len(args) < 2:
                    continue
                embs = embed_model.encode(args)
                for i in range(1, len(embs)):
                    sim = cosine_similarity([embs[i - 1]], [embs[i]])[0][0]
                    rows.append({
                        "protocol": proto,
                        "topic_id": topic_id,
                        "agent": agent_key,
                        "transition": i,  # 1 = round 1→2, 2 = round 2→3, etc.
                        "shift": float(1 - sim),
                    })

    return pd.DataFrame(rows)


def compute_regret_protocol_b(transcript_dir, topics_data, engine):
    """
    Compute cumulative regret for Protocol B agents across rounds.

    Utility is defined as 1 if the agent's stated position matches ground truth,
    0 otherwise. Best fixed action in hindsight = always argue ground truth (utility 1
    per round). Cumulative regret = best_possible_cumulative - actual_cumulative.
    """
    ground_truths = {t["id"]: t["ground_truth"] for t in topics_data}
    rows = []

    for topic_id in range(1, 11):
        filename = Path(transcript_dir) / f"protocol_b_topic_{topic_id}.json"
        if not filename.exists():
            continue
        with open(filename) as f:
            transcript = json.load(f)

        gt = ground_truths.get(topic_id, True)
        rounds_data = transcript.get("rounds", [])

        for agent_key in ["agent1", "agent2"]:
            cumulative_utility = 0.0
            for round_idx, round_data in enumerate(rounds_data):
                arg = round_data.get(agent_key, {}).get("argument", "")
                pos = engine.extract_final_position(arg)
                utility = 1.0 if (pos is not None and pos == gt) else 0.0
                cumulative_utility += utility
                best_possible = float(round_idx + 1)  # 1.0 per round if always correct
                rows.append({
                    "topic_id": topic_id,
                    "agent": agent_key,
                    "round": round_idx + 1,
                    "utility": utility,
                    "cumulative_utility": cumulative_utility,
                    "cumulative_regret": best_possible - cumulative_utility,
                    "ground_truth": gt,
                })

    return pd.DataFrame(rows)


def main():
    """Run analysis and generate all outputs"""
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    Path(FIGURES_DIR).mkdir(exist_ok=True)

    with open("topics.json", "r") as f:
        topics = json.load(f)

    print("Analyzing experiment results...")
    print("-" * 60)

    engine = EvaluationEngine()

    # Compute summary metrics (factual accuracy, mean shift, agreement)
    results_df = evaluate_all_transcripts(TRANSCRIPT_DIR, topics)

    if results_df.empty:
        print("Error: No transcripts found. Run run_experiments.py first.")
        return

    csv_path = Path(RESULTS_DIR) / "metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Metrics saved to {csv_path}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    summary = results_df.groupby("protocol")[
        ["factual_accuracy", "position_shift", "agreement"]
    ].agg(["mean", "std", "min", "max"])
    print(summary)

    protocols = ["A", "B", "C"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # ------------------------------------------------------------------
    # Figure 1: Factual Accuracy per Protocol
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    accuracy_means = [results_df[results_df["protocol"] == p]["factual_accuracy"].mean() for p in protocols]
    accuracy_stds = [results_df[results_df["protocol"] == p]["factual_accuracy"].std() for p in protocols]

    ax.bar(protocols, accuracy_means, yerr=accuracy_stds, capsize=10, color=colors)
    ax.set_ylabel("Factual Accuracy", fontsize=12)
    ax.set_xlabel("Protocol", fontsize=12)
    ax.set_title("Factual Accuracy by Protocol", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.grid(axis="y", alpha=0.3)
    for i, (mean, std) in enumerate(zip(accuracy_means, accuracy_stds)):
        ax.text(i, mean + std + 0.03, f"{mean:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "01_factual_accuracy.png", dpi=300, bbox_inches="tight")
    print("✓ Figure saved: 01_factual_accuracy.png")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 2: Per-Round Position Shift Trajectories (B and C)
    # ------------------------------------------------------------------
    print("Computing per-round position shifts...")
    shift_df = build_per_round_shift_df(TRANSCRIPT_DIR, topics)

    fig, ax = plt.subplots(figsize=(9, 6))
    proto_styles = {
        "B": {"color": "#ff7f0e", "linestyle": "-",  "label": "Protocol B (Sequential, 2 rounds)"},
        "C": {"color": "#2ca02c", "linestyle": "--", "label": "Protocol C (Judge-Mediated, 4 rounds)"},
    }
    for proto, style in proto_styles.items():
        grp = shift_df[shift_df["protocol"] == proto]
        mean_by_t = grp.groupby("transition")["shift"].mean()
        std_by_t = grp.groupby("transition")["shift"].std()
        x = mean_by_t.index.tolist()
        y = mean_by_t.values.tolist()
        yerr = std_by_t.fillna(0).values.tolist()
        x_labels = [f"Rnd {i}→{i + 1}" for i in x]
        ax.errorbar(
            x, y, yerr=yerr,
            marker="o", linewidth=2, capsize=6, markersize=8,
            color=style["color"], linestyle=style["linestyle"], label=style["label"],
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"Rnd {i}→{i+1}" for i in x], fontsize=10)

    ax.set_ylabel("Mean Position Shift (Cosine Distance)", fontsize=12)
    ax.set_xlabel("Round Transition", fontsize=12)
    ax.set_title("Per-Round Argument Position Shift (Protocols B & C)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "02_position_shift.png", dpi=300, bbox_inches="tight")
    print("✓ Figure saved: 02_position_shift.png")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 3: Inter-Agent Agreement at Termination
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    boxes = ax.boxplot(
        [results_df[results_df["protocol"] == p]["agreement"].values for p in protocols],
        tick_labels=protocols,
        patch_artist=True,
        widths=0.6,
    )
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
    print("✓ Figure saved: 03_interagent_agreement.png")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 4: Comparison Matrix (all metrics)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_cols = ["factual_accuracy", "position_shift", "agreement"]
    metric_labels = ["Factual Accuracy", "Position Shift", "Agreement"]

    for ax, metric, label in zip(axes, metric_cols, metric_labels):
        data = [results_df[results_df["protocol"] == p][metric].values for p in protocols]
        bp = ax.boxplot(data, tick_labels=protocols, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(Path(FIGURES_DIR) / "04_all_metrics_boxplot.png", dpi=300, bbox_inches="tight")
    print("✓ Figure saved: 04_all_metrics_boxplot.png")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 5: Cumulative Regret for Protocol B (Regret Minimization Analysis)
    # ------------------------------------------------------------------
    print("Computing cumulative regret for Protocol B (requires LLM calls per round)...")
    regret_df = compute_regret_protocol_b(TRANSCRIPT_DIR, topics, engine)

    if not regret_df.empty:
        regret_csv_path = Path(RESULTS_DIR) / "regret_protocol_b.csv"
        regret_df.to_csv(regret_csv_path, index=False)
        print(f"✓ Regret data saved to {regret_csv_path}")

        # Print regret summary
        print("\nProtocol B — Mean cumulative regret by round:")
        print(regret_df.groupby("round")["cumulative_regret"].agg(["mean", "std"]))

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot mean cumulative regret per round, with ±1 std band
        mean_regret = regret_df.groupby("round")["cumulative_regret"].mean()
        std_regret = regret_df.groupby("round")["cumulative_regret"].std()
        x = mean_regret.index.tolist()

        ax.plot(x, mean_regret.values, marker="o", linewidth=2.5, color="#ff7f0e",
                label="Protocol B mean cumulative regret", markersize=8)
        ax.fill_between(x,
                        mean_regret.values - std_regret.values,
                        mean_regret.values + std_regret.values,
                        alpha=0.2, color="#ff7f0e", label="±1 std")

        # Reference line: zero regret (no-regret learner)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, label="Zero regret (ideal)")

        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Cumulative Regret", fontsize=12)
        ax.set_title("Cumulative Regret — Protocol B (Sequential Debate)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(Path(FIGURES_DIR) / "05_regret_protocol_b.png", dpi=300, bbox_inches="tight")
        print("✓ Figure saved: 05_regret_protocol_b.png")
        plt.close()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
