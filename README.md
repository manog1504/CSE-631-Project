# LLM Debate Protocol Design: A Game-Theoretic Analysis

CSCE 631 Course Project | Texas A&M University | Summer I 2026

## Overview

This project investigates how the structural design of multi-agent LLM debate protocols affects output quality and game-theoretic convergence properties. Three protocols are implemented and evaluated:

- **Protocol A (Simultaneous)**: Models a normal-form game with simultaneous action selection
- **Protocol B (Sequential Rebuttal)**: Models an extensive-form game with perfect recall
- **Protocol C (Judge-Mediated)**: Models a correlated equilibrium structure with mediator signaling

## Project Structure

```
llm_debate_project/
├── config.py                      # API configuration
├── topics.json                    # 10 debate topics with ground truth (5 True / 5 False)
├── run_experiments.py             # Main orchestration script
├── analyze_results.py             # Analysis and figure generation
├── protocols/
│   ├── client.py                  # Anthropic API client wrapper
│   ├── protocol_a.py              # Simultaneous debate
│   ├── protocol_b.py              # Sequential rebuttal
│   └── protocol_c.py             # Judge-mediated debate
├── evaluation/
│   └── metrics.py                 # Factual accuracy, position shift, agreement, regret
├── report/
│   └── report.tex                 # LaTeX report
├── transcripts/                   # JSON logs of all debates (30 files)
├── results/
│   ├── metrics.csv                # Computed metrics table
│   ├── regret_protocol_b.csv      # Per-round regret data for Protocol B
│   └── figures/                   # Generated plots (5 figures)
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install anthropic sentence-transformers scikit-learn pandas matplotlib
```

### 2. Configure API Credentials

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

The key is loaded automatically by `config.py`. Alternatively, set it as an environment variable:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Verify Setup

```bash
cd llm_debate_project
python3 -c "
from protocols.client import call_llm
content, usage = call_llm([{'role': 'user', 'content': 'What is a Nash equilibrium? Answer in one sentence.'}])
print(content)
"
```

## Running Experiments

### Full Experiment Run (30 debates: 3 protocols × 10 topics)

```bash
cd llm_debate_project
python3 run_experiments.py
```

The script skips transcripts that already exist (`skip_completed=True`). Estimated time: 30–45 minutes. Estimated cost: < $0.50 total using Claude Haiku 4.5.

### Single Topic Test

```python
import json
from protocols.protocol_a import run_simultaneous
from protocols.protocol_b import run_sequential
from protocols.protocol_c import run_judge_mediated

with open('topics.json') as f:
    topics = json.load(f)

topic = topics[0]
run_simultaneous(topic['topic'], topic['id'])
run_sequential(topic['topic'], topic['id'])
run_judge_mediated(topic['topic'], topic['id'])
```

## Analysis & Visualization

After experiments complete, generate all metrics and figures:

```bash
cd llm_debate_project
python3 analyze_results.py
```

This produces:
- `results/metrics.csv` — factual accuracy, position shift, agreement per protocol/topic
- `results/regret_protocol_b.csv` — per-round cumulative regret for Protocol B agents
- `results/figures/01_factual_accuracy.png` — bar chart of accuracy by protocol
- `results/figures/02_position_shift.png` — per-round position shift trajectories (B & C)
- `results/figures/03_interagent_agreement.png` — agreement boxplots
- `results/figures/04_all_metrics_boxplot.png` — comprehensive metric comparison
- `results/figures/05_regret_protocol_b.png` — cumulative regret curves for Protocol B

## Report Generation

```bash
cd llm_debate_project/report
pdflatex report.tex
```

## Evaluation Metrics

### Factual Accuracy
- Extracts final position (True/False) from each agent's final argument via LLM classification
- Score = 1 if matches ground truth, 0 otherwise; averaged across both agents per topic

### Position Shift (Cosine Distance)
- Embeds each round's argument using `all-MiniLM-L6-v2`
- Computes cosine distance between consecutive-round embeddings, per agent
- Plotted as a per-round trajectory to test convergence behavior

### Inter-Agent Agreement (Cosine Similarity)
- Embeds both agents' final arguments; measures similarity at termination
- Proxy for convergence to a shared conclusion

### Cumulative Regret (Protocol B only)
- Per-round utility = 1 if agent's position matches ground truth, 0 otherwise
- Best fixed action = always argue ground truth (utility 1 per round)
- Cumulative regret = best_possible_cumulative − actual_cumulative
- Tests whether Protocol B agents exhibit no-regret learning behavior

## Key Concepts from CSCE 631

- **Module 1 (Normal-Form Games)**: Protocol A models simultaneous Nash equilibrium
- **Module 3 (Extensive-Form Games)**: Protocol B models sequential games with perfect recall
- **Module 1 (Correlated Equilibrium)**: Protocol C models CE mediator structure
- **Module 2 (Regret Minimization)**: Protocol B is formally tested for no-regret behavior
- **Module 5 (LLM Agents & Multi-Agent Debate)**: All protocols ground in POMDP framework

## References

- Irving et al. (2018). "AI Safety via Debate." *arXiv*.
- Du et al. (2023). "Improving Factuality and Reasoning in Language Models Through Multi-Agent Debate." *ICML*.
- Cesa-Bianchi & Lugosi (2006). *Prediction, Learning, and Games*. Cambridge University Press.

---

**Author**: Manogna Yadav Narayana Swamy  
**Course**: CSCE 631 — Algorithmic Game Theory meets LLMs  
**Term**: Summer I 2026
