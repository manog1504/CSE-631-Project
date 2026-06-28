"""Protocol C: Judge-Mediated Debate (Correlated Equilibrium)

A third LLM agent (judge) scores each round's arguments and returns scores back to debaters.
Agents see prior scores as signal in subsequent rounds.
Models a correlated equilibrium structure with mediator signaling.
"""

import json
from protocols.client import call_llm
from config import TEMPERATURE, MAX_TOKENS, MAX_ROUNDS, TRANSCRIPT_DIR


def score_round(agent1_arg, agent2_arg, topic_text):
    """
    Judge scores both agents' arguments for accuracy and persuasiveness.

    Returns:
        {
            'agent1_score': float (0-10),
            'agent2_score': float (0-10),
            'agent1_feedback': str,
            'agent2_feedback': str
        }
    """

    judge_prompt = f"""You are an expert judge evaluating debate arguments.

Topic: {topic_text}

Agent 1 (Pro) argument: {agent1_arg}

Agent 2 (Con) argument: {agent2_arg}

Evaluate each argument on a scale of 0-10 based on:
- Factual accuracy
- Logical coherence
- Persuasiveness

Respond in JSON format:
{{
    "agent1_score": <number 0-10>,
    "agent2_score": <number 0-10>,
    "agent1_feedback": "<brief feedback>",
    "agent2_feedback": "<brief feedback>"
}}"""

    response, _ = call_llm(
        messages=[
            {"role": "user", "content": judge_prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    try:
        # Extract JSON from response
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            return scores
    except:
        pass

    # Fallback: return neutral scores
    return {
        "agent1_score": 5.0,
        "agent2_score": 5.0,
        "agent1_feedback": "Unable to parse scores",
        "agent2_feedback": "Unable to parse scores"
    }


def run_judge_mediated(topic_text, topic_id):
    """
    Run judge-mediated debate protocol.

    Each round: both agents argue → judge scores → scores fed back as signal.

    Args:
        topic_text: the debate topic (string)
        topic_id: topic identifier

    Returns:
        {
            'topic_id': int,
            'protocol': 'C',
            'topic': str,
            'rounds': [
                {
                    'round_num': int,
                    'agent1': {'argument': str, 'tokens': int},
                    'agent2': {'argument': str, 'tokens': int},
                    'judge_scores': {...}
                },
                ...
            ],
            'total_tokens': int
        }
    """

    system_prompt = """You are a skilled debater in a structured debate with a judge evaluating each round.
Your goal is to argue convincingly for your assigned position using factual and logical reasoning.
You will receive feedback scores from the judge after each round.
Provide a clear, well-reasoned argument (2-3 sentences) for your round.
Be concise, logical, and persuasive."""

    rounds = []
    total_tokens = 0
    prior_scores = None

    for round_num in range(MAX_ROUNDS):
        # Build per-agent score context
        def score_context(agent_key):
            if not prior_scores:
                return ""
            s = prior_scores[agent_key]
            return (f"\n\nPrevious round scores from the judge:\n"
                    f"- Your score: {s['your_score']}/10 ({s['your_feedback']})\n"
                    f"- Opponent's score: {s['opponent_score']}/10 ({s['opponent_feedback']})")

        # Agent 1 (Pro) argues
        agent1_prompt = f"""Debate topic: {topic_text}

Your position: Pro (the statement is TRUE){score_context('agent1')}

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
            arg1 = "[Failed to generate]"
            tokens1 = {"total_tokens": 0}

        total_tokens += tokens1["total_tokens"]

        # Agent 2 (Con) argues
        agent2_prompt = f"""Debate topic: {topic_text}

Your position: Con (the statement is FALSE){score_context('agent2')}

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
            arg2 = "[Failed to generate]"
            tokens2 = {"total_tokens": 0}

        total_tokens += tokens2["total_tokens"]

        # Judge scores both arguments
        judge_scores = score_round(arg1, arg2, topic_text)

        # Store scores per agent for next round context
        prior_scores = {
            "agent1": {
                "your_score": judge_scores.get("agent1_score", 5.0),
                "your_feedback": judge_scores.get("agent1_feedback", ""),
                "opponent_score": judge_scores.get("agent2_score", 5.0),
                "opponent_feedback": judge_scores.get("agent2_feedback", "")
            },
            "agent2": {
                "your_score": judge_scores.get("agent2_score", 5.0),
                "your_feedback": judge_scores.get("agent2_feedback", ""),
                "opponent_score": judge_scores.get("agent1_score", 5.0),
                "opponent_feedback": judge_scores.get("agent1_feedback", "")
            }
        }

        rounds.append({
            "round_num": round_num + 1,
            "agent1": {
                "argument": arg1,
                "tokens": tokens1["total_tokens"]
            },
            "agent2": {
                "argument": arg2,
                "tokens": tokens2["total_tokens"]
            },
            "judge_scores": judge_scores
        })

    result = {
        "topic_id": topic_id,
        "protocol": "C",
        "topic": topic_text,
        "rounds": rounds,
        "total_tokens": total_tokens
    }

    # Save transcript
    filename = f"{TRANSCRIPT_DIR}/protocol_c_topic_{topic_id}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    return result
