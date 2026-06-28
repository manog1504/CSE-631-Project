"""Evaluation metrics for debate protocols"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EvaluationEngine:
    def __init__(self):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_final_position(self, argument_text):
        """
        Extract final position (True/False) from argument text using LLM.
        Simplified: check for keywords at end of argument.

        Returns: True, False, or None if indeterminate
        """
        if not argument_text or argument_text.startswith("[Failed"):
            return None
        text_lower = argument_text.lower().strip()

        # Simple keyword-based extraction
        if text_lower.endswith("true") or "is true" in text_lower[-20:]:
            return True
        if text_lower.endswith("false") or "is false" in text_lower[-20:]:
            return False
        if "correct" in text_lower[-30:] or "accurate" in text_lower[-30:]:
            return True
        if "incorrect" in text_lower[-30:] or "inaccurate" in text_lower[-30:]:
            return False

        return None  # Indeterminate

    def compute_factual_accuracy(self, transcript, ground_truth):
        """
        Compute factual accuracy metric.

        Protocol A: Check final positions of both agents
        Protocol B/C: Check final arguments of both agents

        Returns: accuracy score (0-1, average of both agents)
        """
        protocol = transcript.get("protocol")
        accuracy_scores = []

        if protocol == "A":
            arg1 = transcript["agent1"]["argument"]
            arg2 = transcript["agent2"]["argument"]

            pos1 = self.extract_final_position(arg1)
            pos2 = self.extract_final_position(arg2)

            if pos1 is not None:
                accuracy_scores.append(1.0 if pos1 == ground_truth else 0.0)
            if pos2 is not None:
                accuracy_scores.append(1.0 if pos2 == ground_truth else 0.0)

        else:  # Protocol B or C
            rounds = transcript.get("rounds", [])
            if rounds:
                # Agent 1 is Pro (True), Agent 2 is Con (False)
                last_round = rounds[-1]

                arg1 = last_round.get("agent1", {}).get("argument", "")
                arg2 = last_round.get("agent2", {}).get("argument", "")

                pos1 = self.extract_final_position(arg1)
                pos2 = self.extract_final_position(arg2)

                if pos1 is not None:
                    accuracy_scores.append(1.0 if pos1 == True else 0.0)
                if pos2 is not None:
                    accuracy_scores.append(1.0 if pos2 == False else 0.0)

        return np.mean(accuracy_scores) if accuracy_scores else 0.5

    def compute_position_shift(self, transcript):
        """
        Compute average position shift (cosine distance between consecutive rounds).

        For Protocol A: N/A (only 1 round) → return 0
        For Protocol B/C: average distance between agent's embeddings across rounds

        Returns: list of shifts per agent, overall mean shift
        """
        protocol = transcript.get("protocol")

        if protocol == "A":
            return 0.0, []

        rounds = transcript.get("rounds", [])
        if len(rounds) < 2:
            return 0.0, []

        shifts = {"agent1": [], "agent2": []}

        # Agent 1 position shifts
        agent1_args = [r["agent1"]["argument"] for r in rounds if "agent1" in r and r["agent1"].get("argument")]
        agent1_args = [a for a in agent1_args if not a.startswith("[Failed")]
        if len(agent1_args) > 1:
            agent1_embs = self.embed_model.encode(agent1_args)
            for i in range(1, len(agent1_embs)):
                sim = cosine_similarity([agent1_embs[i-1]], [agent1_embs[i]])[0][0]
                distance = 1 - sim  # Convert similarity to distance
                shifts["agent1"].append(distance)

        # Agent 2 position shifts
        agent2_args = [r["agent2"]["argument"] for r in rounds if "agent2" in r and r["agent2"].get("argument")]
        agent2_args = [a for a in agent2_args if not a.startswith("[Failed")]
        if len(agent2_args) > 1:
            agent2_embs = self.embed_model.encode(agent2_args)
            for i in range(1, len(agent2_embs)):
                sim = cosine_similarity([agent2_embs[i-1]], [agent2_embs[i]])[0][0]
                distance = 1 - sim
                shifts["agent2"].append(distance)

        all_shifts = shifts["agent1"] + shifts["agent2"]
        mean_shift = np.mean(all_shifts) if all_shifts else 0.0

        return mean_shift, shifts

    def compute_interagent_agreement(self, transcript):
        """
        Compute inter-agent agreement at termination (cosine similarity of final arguments).

        Protocol A: similarity between agent1 and agent2's arguments
        Protocol B/C: similarity between agents' final-round arguments

        Returns: agreement score (0-1, where 1 = perfect agreement)
        """
        protocol = transcript.get("protocol")

        if protocol == "A":
            arg1 = transcript["agent1"]["argument"]
            arg2 = transcript["agent2"]["argument"]
        else:
            rounds = transcript.get("rounds", [])
            if not rounds:
                return 0.5
            last_round = rounds[-1]
            arg1 = last_round.get("agent1", {}).get("argument", "")
            arg2 = last_round.get("agent2", {}).get("argument", "")

        if not arg1 or not arg2:
            return 0.5

        embs = self.embed_model.encode([arg1, arg2])
        similarity = cosine_similarity([embs[0]], [embs[1]])[0][0]

        # Clip to [0, 1] range
        return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize from [-1, 1] to [0, 1]


def evaluate_all_transcripts(transcript_dir, topics_data):
    """
    Load all transcripts and compute metrics across all protocols.

    Args:
        transcript_dir: directory with transcript JSON files
        topics_data: list of topic dicts with 'id' and 'ground_truth'

    Returns:
        pd.DataFrame with columns: protocol, topic_id, factual_accuracy, position_shift, agreement
    """
    import pandas as pd

    engine = EvaluationEngine()
    results = []

    # Create topic lookup
    ground_truths = {t["id"]: t["ground_truth"] for t in topics_data}

    # Scan all transcripts
    for proto in ["A", "B", "C"]:
        for topic_id in range(1, 11):
            filename = Path(transcript_dir) / f"protocol_{proto.lower()}_topic_{topic_id}.json"

            if not filename.exists():
                print(f"Warning: {filename} not found")
                continue

            with open(filename, "r") as f:
                transcript = json.load(f)

            ground_truth = ground_truths.get(topic_id, True)

            acc = engine.compute_factual_accuracy(transcript, ground_truth)
            shift, _ = engine.compute_position_shift(transcript)
            agreement = engine.compute_interagent_agreement(transcript)

            results.append({
                "protocol": proto,
                "topic_id": topic_id,
                "factual_accuracy": acc,
                "position_shift": shift,
                "agreement": agreement
            })

    return pd.DataFrame(results)
