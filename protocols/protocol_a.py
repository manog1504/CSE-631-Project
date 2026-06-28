"""Protocol A: Simultaneous Debate (Normal-Form Game)

Both agents submit arguments independently with no knowledge of the opponent's position.
Models a normal-form game with simultaneous action selection.
"""

import json
from protocols.client import call_llm
from config import TEMPERATURE, MAX_TOKENS, TRANSCRIPT_DIR


def run_simultaneous(topic_text, topic_id):
    """
    Run simultaneous debate protocol for a given topic.

    Args:
        topic_text: the debate topic (string)
        topic_id: topic identifier

    Returns:
        {
            'topic_id': int,
            'protocol': 'A',
            'topic': str,
            'agent1': {'position': 'Pro', 'argument': str, 'tokens': int},
            'agent2': {'position': 'Con', 'argument': str, 'tokens': int}
        }
    """

    system_prompt = """You are a skilled debater participating in a structured debate.
Your goal is to argue convincingly for your assigned position using factual and logical reasoning.
Provide a single, well-reasoned argument (2-3 sentences) supporting your position.
Be concise and persuasive."""

    # Agent 1 argues PRO
    agent1_prompt = f"""Debate topic: {topic_text}

Your position: PRO (argue that the statement is TRUE)

Please provide your argument:"""

    arg1, tokens1 = call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": agent1_prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    if arg1 is None:
        arg1 = "[Failed to generate argument]"
        tokens1 = {"total_tokens": 0}

    # Agent 2 argues CON
    agent2_prompt = f"""Debate topic: {topic_text}

Your position: CON (argue that the statement is FALSE)

Please provide your argument:"""

    arg2, tokens2 = call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": agent2_prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    if arg2 is None:
        arg2 = "[Failed to generate argument]"
        tokens2 = {"total_tokens": 0}

    result = {
        "topic_id": topic_id,
        "protocol": "A",
        "topic": topic_text,
        "agent1": {
            "position": "Pro",
            "argument": arg1,
            "tokens": tokens1["total_tokens"]
        },
        "agent2": {
            "position": "Con",
            "argument": arg2,
            "tokens": tokens2["total_tokens"]
        }
    }

    # Save transcript
    filename = f"{TRANSCRIPT_DIR}/protocol_a_topic_{topic_id}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    return result
