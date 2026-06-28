"""Orchestrate all debate experiments"""

import json
import time
from pathlib import Path
from protocols.protocol_a import run_simultaneous
from protocols.protocol_b import run_sequential
from protocols.protocol_c import run_judge_mediated
from config import TRANSCRIPT_DIR


def run_all_experiments(topics_file="topics.json", skip_completed=True):
    """
    Run all three protocols across all 10 topics.

    Args:
        topics_file: path to topics.json
        skip_completed: if True, skip transcripts that already exist
    """
    # Load topics
    with open(topics_file, "r") as f:
        topics = json.load(f)

    total_experiments = len(topics) * 3  # 3 protocols per topic
    completed = 0

    print(f"Running {total_experiments} experiments (3 protocols × {len(topics)} topics)")
    print(f"Transcripts will be saved to {TRANSCRIPT_DIR}/")
    print("-" * 60)

    for topic in topics:
        topic_id = topic["id"]
        topic_text = topic["topic"]

        print(f"\nTopic {topic_id}: {topic_text}")
        print(f"Ground Truth: {topic['ground_truth']}")

        # Protocol A
        proto_a_file = Path(TRANSCRIPT_DIR) / f"protocol_a_topic_{topic_id}.json"
        if proto_a_file.exists() and skip_completed:
            print("  [A] Skipping (already completed)")
        else:
            print("  [A] Running simultaneous debate...", end="", flush=True)
            try:
                result_a = run_simultaneous(topic_text, topic_id)
                print(" ✓")
                completed += 1
            except Exception as e:
                print(f" ✗ Error: {e}")

        time.sleep(1)  # Small delay between calls

        # Protocol B
        proto_b_file = Path(TRANSCRIPT_DIR) / f"protocol_b_topic_{topic_id}.json"
        if proto_b_file.exists() and skip_completed:
            print("  [B] Skipping (already completed)")
        else:
            print("  [B] Running sequential rebuttal...", end="", flush=True)
            try:
                result_b = run_sequential(topic_text, topic_id)
                print(" ✓")
                completed += 1
            except Exception as e:
                print(f" ✗ Error: {e}")

        time.sleep(1)

        # Protocol C
        proto_c_file = Path(TRANSCRIPT_DIR) / f"protocol_c_topic_{topic_id}.json"
        if proto_c_file.exists() and skip_completed:
            print("  [C] Skipping (already completed)")
        else:
            print("  [C] Running judge-mediated debate...", end="", flush=True)
            try:
                result_c = run_judge_mediated(topic_text, topic_id)
                print(" ✓")
                completed += 1
            except Exception as e:
                print(f" ✗ Error: {e}")

        time.sleep(1)

    print("\n" + "=" * 60)
    print(f"Experiment run complete. {completed}/{total_experiments} protocols executed.")
    print(f"Transcripts saved to {TRANSCRIPT_DIR}/")


if __name__ == "__main__":
    run_all_experiments()
