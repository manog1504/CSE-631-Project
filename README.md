# LLM Debate Protocol Design: A Game-Theoretic Analysis

CSCE 631 Course Project | Texas A&M University | Summer I 2026

## Overview

This project investigates how the structural design of multi-agent LLM debate protocols affects output quality and game-theoretic convergence properties. We implement and evaluate three protocols:

- **Protocol A (Simultaneous)**: Models a normal-form game with simultaneous action selection
- **Protocol B (Sequential Rebuttal)**: Models an extensive-form game with perfect recall
- **Protocol C (Judge-Mediated)**: Models a correlated equilibrium structure with mediator signaling

## Project Structure

```
llm_debate_project/
├── config.py                      # TAMU API configuration
├── topics.json                    # 10 debate topics with ground truth
├── run_experiments.py             # Main orchestration script
├── analyze_results.py             # Analysis and figure generation
├── protocols/
│   ├── client.py                  # TAMU API client wrapper
│   ├── protocol_a.py              # Simultaneous debate
│   ├── protocol_b.py              # Sequential rebuttal
│   └── protocol_c.py              # Judge-mediated debate
├── evaluation/
│   └── metrics.py                 # Evaluation metrics engine
├── report/
│   └── report.tex                 # LaTeX report template
├── transcripts/                   # JSON logs of all debates
├── results/
│   ├── metrics.csv                # Computed metrics table
│   └── figures/                   # Generated plots
└── README.md                      # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install openai sentence-transformers scikit-learn pandas matplotlib
```

### 2. Configure API Credentials

The `config.py` file is pre-configured with:
- **Base URL**: `https://chat.tamu.ai/api`
- **API Key**: Shared course key (in config.py)
- **CF_Authorization Cookie**: Your personal Cloudflare Access cookie (expires every 24 hours)

**Important**: If the cookie expires, log into `https://chat.tamu.ai`, extract a fresh `CF_Authorization` cookie via browser DevTools (see TAMU-API-Guide.pdf), and update `config.py`.

### 3. Verify Setup

Test the API connection:

```bash
cd llm_debate_project
python -c "from protocols.client import call_llm; content, usage = call_llm([{'role': 'user', 'content': 'What is a Nash equilibrium? Answer in one sentence.'}]); print(content)"
```

If you see a one-sentence answer about Nash equilibrium, your setup is working.

## Running Experiments

### Quick Start (Single Topic)

```bash
cd llm_debate_project
python -c "
import json
from protocols.protocol_a import run_simultaneous
from protocols.protocol_b import run_sequential
from protocols.protocol_c import run_judge_mediated

with open('topics.json') as f:
    topics = json.load(f)

topic = topics[0]
print(f'Testing with: {topic[\"topic\"]}')
print('Protocol A...'); run_simultaneous(topic['topic'], topic['id'])
print('Protocol B...'); run_sequential(topic['topic'], topic['id'])
print('Protocol C...'); run_judge_mediated(topic['topic'], topic['id'])
print('Done!')
"
```

### Full Experiment Run (All 3 Protocols × 10 Topics = 30 Debates)

```bash
cd llm_debate_project
python run_experiments.py
```

**Estimated Time**: ~30-45 minutes (depends on API latency)

**Cost**: ~$1-2 total (within $5/day budget)

The script will log progress and save transcripts to `transcripts/protocol_*.json`.

## Analysis & Visualization

After experiments complete, generate metrics and figures:

```bash
cd llm_debate_project
python analyze_results.py
```

This produces:
- `results/metrics.csv` — factual accuracy, position shift, agreement scores
- `results/figures/01_factual_accuracy.png` — bar chart of accuracy by protocol
- `results/figures/02_position_shift.png` — convergence curves (B & C only)
- `results/figures/03_interagent_agreement.png` — agreement boxplots
- `results/figures/04_all_metrics_boxplot.png` — comprehensive metric comparison

## Report Generation

Compile the LaTeX report:

```bash
cd llm_debate_project/report
pdflatex report.tex
```

This generates `report.pdf` (4 pages) with results, game-theoretic framework, and conclusions.

## Evaluation Metrics

### Factual Accuracy
- Extracts final position (True/False) from each agent's final argument
- Score = 1 if matches ground truth, 0 otherwise
- Averaged across both agents and all topics per protocol

### Position Shift (Cosine Distance)
- Embeds each round's argument using `all-MiniLM-L6-v2`
- Computes cosine distance between consecutive rounds
- Lower = more stable, Higher = more adaptation
- Only computed for Protocols B & C (which have multiple rounds)

### Inter-Agent Agreement (Cosine Similarity)
- Embeds both agents' final arguments
- Measures similarity: 1 = perfect agreement, 0 = orthogonal
- Proxy for convergence to a shared conclusion

## Budget & Cost-Saving Tips

- **Daily Cap**: $5/day per student (enforced via CF_Authorization cookie)
- **Cost Estimate**: ~$0.10-0.20 per debate (depends on argument length)
- **Savings**:
  - Cache intermediate results (transcripts are automatically saved)
  - Use `skip_completed=True` in `run_experiments.py` to skip already-run experiments
  - Consider using `protected.Claude-Haiku-4.5` or `protected.gpt-5-mini` for high-volume testing (cheaper)

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `401 Unauthorized` | Cookie expired | Log into chat.tamu.ai, extract fresh CF_Authorization cookie |
| `"model not found"` | Wrong model name | Check model name includes `protected.` prefix |
| `429 Too Many Requests` | Rate limiting | Add `time.sleep(1)` between calls |
| Empty transcript | API call failed | Check internet connection; see transcript file for error details |

## Key Concepts from CSCE 631

- **Module 1 (Normal-Form Games)**: Protocol A models simultaneous Nash equilibrium
- **Module 3 (Extensive-Form Games)**: Protocol B models sequential games with perfect recall
- **Module 1 (Correlated Equilibrium)**: Protocol C models CE mediator structure
- **Module 2 (Regret Minimization)**: Protocol B analyzes no-regret convergence
- **Module 5 (LLM Agents & Multi-Agent Debate)**: All protocols ground in POMDP framework

## References

- Irving et al. (2018). "AI Safety via Debate." *arXiv*.
- Du et al. (2023). "Improving Factuality and Reasoning in Language Models Through Multi-Agent Debate." *ICML*.
- Course materials: CSCE 631 Modules 1-5.

## Contact

For questions about the TAMU API or course project, see the CSCE 631 course page.

---

**Author**: Manogna Yadav Narayana Swamy  
**Course**: CSCE 631 — Algorithmic Game Theory meets LLMs  
**Term**: Summer I 2026
