# Copy to config.py and fill in. config.py holds secrets; do not commit it.
GOOGLE_API_KEY = "unused-when-claude-is-configured"  # main.py still calls genai.configure()
GOOGLE_MODEL_NAME = "gemini-1.5-flash"

# Anthropic Claude: when a key is set it is used for category, role and synonym calls.
# The key can also come from the ANTHROPIC_API_KEY environment variable.
ANTHROPIC_API_KEY = None
ANTHROPIC_MODEL_NAME = "claude-sonnet-5-5"

OUTPUT_CSV_PATH = "./meeting_analysis_results_belbin_only.csv"
LOG_LEVEL = "INFO"
DELAY_SECONDS = 2
MAX_RETRIES = 3
