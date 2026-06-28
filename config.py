"""TAMU API Configuration for CSCE 631 Debate Project"""

# TAMU API Details
TAMU_BASE_URL = "https://chat.tamu.ai/api"
TAMU_API_KEY = "sk-0183ed7c1c8e47c0a8f4d1e75161aa6d"

# Cloudflare Authorization Cookie (personal to your NetID, expires every 24h)
# Replace with your fresh CF_Authorization cookie if this expires
CF_AUTHORIZATION_COOKIE = "CF_Authorization=eyJ..."  # Replace with your CF_Authorization cookie from chat.tamu.ai

# Model Configuration
MODEL = "protected.Claude Sonnet 4.5"  # Best balance of quality and cost
TEMPERATURE = 1  # Claude thinking models require temperature=1
MAX_TOKENS = 16384  # Claude thinking models require max_tokens>=16384
MAX_ROUNDS = 4  # Rounds for Protocol B and C

# Debate Protocol Settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Output Directories
TRANSCRIPT_DIR = "transcripts"
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
REPORT_DIR = "report"
