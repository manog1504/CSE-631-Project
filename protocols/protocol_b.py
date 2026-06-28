"""Protocol B: Sequential with Rebuttal (Extensive-Form Game)

Agents alternate arguments across multiple rounds with full history visibility.
Each agent sees prior arguments before responding.
Models an extensive-form game with perfect recall.
"""

import json
from protocols.client import call_llm
from config import TEMPERATURE, MAX_TOKENS, MAX_ROUNDS, TRANSCRIPT_DIR


def run_sequential(topic_text, topic_id):
    """
    Run sequential debate protocol with alternating rebuttals.

    Args:
        topic_text: the debate topic (string)
        topic_id: topic identifier

    Returns:
        {
            'topic_id': int,
            'protocol': 'B',
            'topic': str,
            'rounds': [
                {
                    'round_num': int,
                    'agent': str (Agent 1 or Agent 2),
                    'argument': str,
                    'tokens': int
                },
                ...
            ],
            'total_tokens': int
        }
    """

    system_prompt = """You are a skilled debater in a structured debate with alternating arguments.
Your goal is to argue convincingly for your assigned position using factual and logical reasoning.
You will see the opponent's previous arguments and can directly address and rebut them.
Provide a clear, well-reasoned argument (2-3 sentences) for your round.
Be concise, logical, and persuasive."""

    history = []
    rounds = []
    total_tokens = 0

    num_rounds = MAX_ROUNDS // 2  # Each "round" has both agents argue once

    for round_num in range(num_rounds):
        # Build context from history
        history_text = ""
        if history:
            history_text = "\n\nPrevious arguments in this debate:\n"
            for i, (prev_agent, prev_arg) in enumerate(history, 1):
                history_text += f"- Agent {prev_agent} (Turn {i}): {prev_arg}\n"

        # Agent 1 argues PRO
        agent1_prompt = f"""Debate topic: {topic_text}

Your position: Pro (arguing the statement is TRUE){history_text}

Please provide your argument for Round {round_num + 1}:"""

        arg1, tokens1 = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": agent1_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        if arg1 is None:
            print(f"Failed to get argument from Agent 1 in round {round_num + 1}")
            arg1 = "[Failed to generate argument]"
            tokens1 = {"total_tokens": 0}

        total_tokens += tokens1["total_tokens"]
        history.append((1, arg1))

        # Agent 2 argues CON (sees Agent 1's response)
        history_text2 = "\n\nPrevious arguments in this debate:\n"
        for i, (prev_agent, prev_arg) in enumerate(history, 1):
            history_text2 += f"- Agent {prev_agent} (Turn {i}): {prev_arg}\n"

        agent2_prompt = f"""Debate topic: {topic_text}

Your position: Con (arguing the statement is FALSE){history_text2}

Please provide your argument for Round {round_num + 1}:"""

        arg2, tokens2 = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": agent2_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        if arg2 is None:
            print(f"Failed to get argument from Agent 2 in round {round_num + 1}")
            arg2 = "[Failed to generate argument]"
            tokens2 = {"total_tokens": 0}

        total_tokens += tokens2["total_tokens"]
        history.append((2, arg2))

        rounds.append({
            "round_num": round_num + 1,
            "agent1": {
                "argument": arg1,
                "tokens": tokens1["total_tokens"]
            },
            "agent2": {
                "argument": arg2,
                "tokens": tokens2["total_tokens"]
            }
        })

    result = {
        "topic_id": topic_id,
        "protocol": "B",
        "topic": topic_text,
        "rounds": rounds,
        "total_tokens": total_tokens
    }

    # Save transcript
    filename = f"{TRANSCRIPT_DIR}/protocol_b_topic_{topic_id}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    return result
