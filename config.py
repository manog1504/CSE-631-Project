"""Anthropic API Configuration for CSCE 631 Debate Project"""

import os

# Load from .env file if present
try:
    with open(os.path.join(os.path.dirname(__file__), ".env")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
except FileNotFoundError:
    pass

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Model Configuration
MODEL = "claude-haiku-4-5"
TEMPERATURE = 0.7
MAX_TOKENS = 1024
MAX_ROUNDS = 4

# Debate Protocol Settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Output Directories
TRANSCRIPT_DIR = "transcripts"
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
REPORT_DIR = "report"
