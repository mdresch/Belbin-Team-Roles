# utils.py
import google.generativeai as genai
import json
import logging
import time
import os
import sys
from openai import OpenAI # Client class
import openai # <<< ADDED: Main library for exception types
import requests # For local server check

# --- Configuration Import ---
# Ensure logging is minimally configured BEFORE the first potential error log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("DEBUG: utils.py attempting to import from config...") # Add debug print

try:
    # Import config module directly
    import config
    print("DEBUG: utils.py successfully imported 'config' module.")

    # Access variables via attributes, using getattr for optional ones
    DELAY_SECONDS = config.DELAY_SECONDS
    MAX_RETRIES = config.MAX_RETRIES
    GOOGLE_MODEL_NAME = config.GOOGLE_MODEL_NAME

    # Access local LLM vars defensively
    LOCAL_LLM_URL = getattr(config, 'LOCAL_LLM_URL', None)
    LOCAL_LLM_MODEL_NAME = getattr(config, 'LOCAL_LLM_MODEL_NAME', None)
    LOCAL_LLM_API_KEY = getattr(config, 'LOCAL_LLM_API_KEY', 'ollama') # Default API key

    LOCAL_MODEL_CONFIGURED = bool(LOCAL_LLM_URL and LOCAL_LLM_MODEL_NAME)
    print(f"DEBUG: utils.py accessed config variables. LOCAL_MODEL_CONFIGURED={LOCAL_MODEL_CONFIGURED}")
    if LOCAL_MODEL_CONFIGURED:
        print(f"DEBUG: Local LLM URL: {LOCAL_LLM_URL}, Model: {LOCAL_LLM_MODEL_NAME}")
    else:
        print("DEBUG: Local LLM not configured or variables missing in config.py.")

    # Access Optional Fine-tuning Vars if needed later
    # FINE_TUNED_MODEL_PATH = getattr(config, 'FINE_TUNED_MODEL_PATH', None)
    # ... etc.

except ImportError:
    # This catches if 'config.py' itself cannot be found
    logging.critical(f"CRITICAL: utils.py failed to import the 'config' module. Check if config.py exists. Exiting.")
    print(f"CRITICAL: utils.py failed to import the 'config' module. Check if config.py exists. Exiting.")
    sys.exit(1)
except AttributeError as e:
    # This catches if a REQUIRED variable (like DELAY_SECONDS) is missing
    logging.critical(f"CRITICAL: utils.py failed to access required attribute from config module: {e}. Exiting.")
    print(f"CRITICAL: utils.py failed to access required attribute from config module: {e}. Exiting.")
    sys.exit(1)
except Exception as e:
    # Catch any other unexpected errors during config access
    logging.critical(f"CRITICAL: Unexpected error accessing config variables in utils.py: {e}", exc_info=True)
    print(f"CRITICAL: Unexpected error accessing config variables in utils.py: {e}")
    sys.exit(1)


# --- Import prompts safely ---
print("DEBUG: utils.py attempting to import from prompts...")
try:
    from prompts import FIRST_LEVEL_PROMPT, BELBIN_ROLE_PROMPTS, SYNONYM_PROMPT, TONE_PROMPT
    print("DEBUG: utils.py successfully imported prompts.")
except ImportError as e:
    logging.critical(f"CRITICAL: utils.py failed to import required names from prompts.py: {e}. Exiting.")
    print(f"CRITICAL: utils.py failed to import required names from prompts.py: {e}. Exiting.")
    sys.exit(1)

# <<< REMOVED DUPLICATE PROMPT IMPORT BLOCK >>>

# --- Optional dependencies (Transformers/PyTorch) ---
# ... (Keep this section if you still have the unused fine-tuning/sentiment functions) ...
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch
    import numpy as np
    transformers_available = True
except ImportError:
    transformers_available = False
    logging.warning("Transformers or PyTorch not installed. Fine-tuned model and pretrained sentiment analysis features will be disabled.")

# --- Initialize Optional Components ---
# ... (Keep sentiment_pipeline init if needed) ...
sentiment_pipeline = None
if transformers_available:
     try:
         sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
         logging.info("Initialized general sentiment pipeline.")
     except Exception as e:
         logging.warning(f"Could not load general sentiment pipeline: {e}")
         sentiment_pipeline = None

# ... (Keep fine-tuned model components init if needed) ...
tokenizer = None
model = None
label_map_global = {}
inverted_label_map = {}
model_loaded = False
# ... (Keep load_fine_tuned_model function if needed) ...


# --- ADD Local LLM Client Initialization ---
# (Code is same as before, relies on variables defined above)
local_client = None
local_model_available = False
if LOCAL_MODEL_CONFIGURED:
    try:
        ping_url = LOCAL_LLM_URL.replace("/v1", "")
        try:
            # Uses 'requests' library (import added at top)
            response = requests.get(ping_url, timeout=3) # Increased timeout slightly
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            logging.info(f"Local LLM server appears reachable at {ping_url}")
            local_client = OpenAI(
                base_url=LOCAL_LLM_URL,
                api_key=LOCAL_LLM_API_KEY,
            )
            local_model_available = True
            logging.info(f"Initialized OpenAI client for local LLM: {LOCAL_LLM_URL}, Model: {LOCAL_LLM_MODEL_NAME}")
        except requests.exceptions.RequestException as e:
             logging.error(f"Local LLM server unreachable or returned error at {ping_url}: {e}. Local model calls disabled.")
             local_model_available = False
    except Exception as e:
        logging.error(f"Failed to initialize local LLM client: {e}", exc_info=True)
        local_model_available = False
# --- End Local LLM Client Initialization ---


# --- API Calling Functions ---

# call_gemini_api remains the same as previously corrected version
def call_gemini_api(prompt_template, sentence, model_name=GOOGLE_MODEL_NAME, delay=DELAY_SECONDS, max_retries=MAX_RETRIES):
    # ... (function body) ...
    retries = 0
    current_delay = delay
    try:
        formatted_prompt = prompt_template.format(sentence=sentence)
    except KeyError:
        logging.error(f"Prompt template missing '{{sentence}}' placeholder: {prompt_template[:100]}...")
        return None, "PROMPT_FORMAT_ERROR" # Return error type
    except Exception as fmt_err:
         logging.error(f"Error formatting prompt: {fmt_err}")
         return None, "PROMPT_FORMAT_ERROR" # Return error type

    logging.debug(f"Calling Gemini API. Retry {retries}/{max_retries}. Prompt: {formatted_prompt[:200]}...")

    while retries < max_retries:
        try:
            model_instance = genai.GenerativeModel(model_name)
            response = model_instance.generate_content(formatted_prompt)

            # --- Enhanced Response Checking ---
            finish_reason = "UNKNOWN"
            block_reason = None
            safety_ratings = []
            try:
                if response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason.name
                    if hasattr(candidate, 'safety_ratings'):
                         safety_ratings = candidate.safety_ratings
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason.name
                    finish_reason = f"BLOCKED_{block_reason}"
            except (AttributeError, IndexError, ValueError) as resp_err:
                 logging.warning(f"Could not fully parse response metadata: {resp_err}")

            if block_reason:
                logging.error(f"Gemini API call blocked for prompt. Reason: {block_reason}. Prompt: {formatted_prompt[:200]}...")
                return None, "SAFETY_BLOCKED_PROMPT"
            if finish_reason == "SAFETY":
                logging.warning(f"Gemini response finished due to SAFETY. Ratings: {safety_ratings}. Prompt: {formatted_prompt[:200]}...")
                return None, "SAFETY_BLOCKED_RESPONSE"
            if finish_reason not in ["STOP", "MAX_TOKENS"]:
                 logging.warning(f"Gemini response finished with unexpected reason: {finish_reason}. Prompt: {formatted_prompt[:200]}...")

            try:
                if not response.parts:
                     logging.warning(f"Gemini response has no parts (finish_reason: {finish_reason}). Returning empty text.")
                     return None, "EMPTY_RESPONSE"
                response_text = response.text
                logging.debug(f"Gemini API Raw Response: {response_text}")
                return response_text, "SUCCESS"
            except ValueError as e:
                 logging.error(f"Error accessing response.text (finish_reason: {finish_reason}, block_reason: {block_reason}): {e}")
                 if block_reason: return None, f"SAFETY_BLOCKED_{block_reason}"
                 if finish_reason == "SAFETY": return None, "SAFETY_BLOCKED_RESPONSE"
                 return None, "ACCESS_ERROR"
            except Exception as text_err:
                 logging.error(f"Unexpected error accessing response.text: {text_err}")
                 return None, "ACCESS_ERROR"

        except Exception as api_err:
            retries += 1
            logging.warning(f"Gemini API Error/Exception (Attempt {retries}/{max_retries}): {api_err}")
            is_rate_limit = "429" in str(api_err)
            if retries >= max_retries:
                logging.error(f"Max retries exceeded for API call. Prompt: {formatted_prompt[:200]}...")
                return None, "API_MAX_RETRIES"
            sleep_time = current_delay * (1.5 ** (retries - 1))
            sleep_time *= (1 + (0.4 * (time.time() % 1) - 0.2))
            sleep_time = max(1, sleep_time)
            if is_rate_limit:
                 sleep_time = max(sleep_time, 5)
                 logging.warning(f"Rate limit likely hit. Retrying in {sleep_time:.2f} seconds.")
            else:
                 logging.info(f"Retrying in {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
    return None, "UNKNOWN_API_FAILURE"


# call_local_llm_api includes the corrected exception handling
def call_local_llm_api(prompt_template, sentence, model_name=LOCAL_LLM_MODEL_NAME, max_retries=MAX_RETRIES, delay=0.5):
    """Calls the local LLM API (OpenAI compatible) with retries."""
    if not local_model_available or not local_client:
        logging.error("Attempted to call local LLM, but client is not available/configured.")
        return None, "LOCAL_MODEL_UNAVAILABLE"

    retries = 0
    current_delay = delay
    try:
        formatted_prompt = prompt_template.format(sentence=sentence)
    except KeyError:
        logging.error(f"Local LLM prompt template missing '{{sentence}}' placeholder: {prompt_template[:100]}...")
        return None, "PROMPT_FORMAT_ERROR"
    except Exception as fmt_err:
        logging.error(f"Error formatting local LLM prompt: {fmt_err}")
        return None, "PROMPT_FORMAT_ERROR"

    logging.debug(f"Calling Local LLM API. Retry {retries}/{max_retries}. Model: {model_name}. Prompt: {formatted_prompt[:200]}...")

    while retries < max_retries:
        try:
            response = local_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert assistant. Respond ONLY in the requested format."},
                    {"role": "user", "content": formatted_prompt}
                ],
                model=model_name,
                max_tokens=1000, # Reduced max_tokens
                temperature=0.6,
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                response_text = response.choices[0].message.content
                logging.debug(f"Local LLM API Raw Response: {response_text}")
                return response_text, "SUCCESS"
            else:
                logging.warning(f"Local LLM response structure unexpected or content empty. Choices: {response.choices}")
                raise ValueError("Empty or invalid response structure")

        except Exception as api_err:
            retries += 1
            logging.warning(f"Local LLM API Error (Attempt {retries}/{max_retries}): {api_err}")
            if retries >= max_retries:
                logging.error(f"Max retries exceeded for Local LLM call. Prompt: {formatted_prompt[:200]}...")
                # --- CORRECTED EXCEPTION CHECK ---
                # Check specific errors using the imported 'openai' module name
                if isinstance(api_err, openai.APIConnectionError):
                    return None, "LOCAL_MODEL_CONNECTION_ERROR"
                elif isinstance(api_err, openai.NotFoundError):
                     return None, "LOCAL_MODEL_NOT_FOUND"
                # --- END CORRECTED ---
                return None, "LOCAL_MODEL_API_ERROR" # Fallback generic error

            sleep_time = current_delay * (1.5 ** (retries - 1))
            sleep_time = max(0.5, sleep_time)
            logging.info(f"Retrying local LLM call in {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)

    return None, "LOCAL_MODEL_UNKNOWN_FAILURE"


# parse_json_response remains the same
def parse_json_response(response_text):
    # ... (function body) ...
    if not response_text:
        logging.warning("Attempted to parse empty response text.")
        return None
    try:
        cleaned_text = response_text.strip().lstrip('```json').rstrip('```').strip()
        if not cleaned_text:
             logging.warning("Response text was empty after cleaning markdown.")
             return None
        data = json.loads(cleaned_text)
        if not isinstance(data, dict):
             logging.warning(f"Parsed JSON is not a dictionary. Type: {type(data)}. Response: {cleaned_text}")
             return None
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON: {e}. Response text: '{response_text}'")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing JSON: {e}. Response text: '{response_text}'")
        return None


# _handle_api_call_result includes LOCAL_MODEL_NOT_FOUND handling
def _handle_api_call_result(response_text, status_code, step_name, sentence_snippet):
    """Helper function to handle results from API calls."""
    if status_code != "SUCCESS":
        logging.error(f"API call failed for {step_name} step ({status_code}) for: '{sentence_snippet}'")
        if "SAFETY_BLOCKED" in status_code:
            return "SAFETY_BLOCKED", None, f"API Call Blocked ({status_code})"
        elif status_code == "LOCAL_MODEL_NOT_FOUND": # Check specific status
             return "LOCAL_API_ERROR", None, f"Local LLM Error ({status_code})"
        elif "LOCAL_MODEL" in status_code:
             return "LOCAL_API_ERROR", None, f"Local LLM Error ({status_code})"
        else:
            return "APIERROR", None, f"API Call Failed ({status_code})"

    response_json = parse_json_response(response_text)
    if response_json is None:
        logging.warning(f"JSON parsing failed for {step_name} step for: '{sentence_snippet}'. Raw Text: '{response_text}'")
        return "JSONERROR", None, "JSON Parse Error"

    return None, response_json, None # Success


# get_llm_category remains the same (uses Gemini)
def get_llm_category(sentence, delay=DELAY_SECONDS, max_retries=MAX_RETRIES):
    # ... (function body) ...
    logging.info(f"Getting LLM category for: '{sentence[:50]}...'")
    sentence_snippet = sentence[:50] + '...'
    response_text, status_code = call_gemini_api(FIRST_LEVEL_PROMPT, sentence, delay=delay, max_retries=max_retries)
    error_label, result_data, error_context = _handle_api_call_result(response_text, status_code, "category", sentence_snippet)
    if error_label:
        return error_label, [], error_context
    response_json = result_data
    category_label = response_json.get("category", "Unknown").lower()
    category_keywords = response_json.get("keywords", [])
    if not isinstance(category_keywords, list):
        logging.warning(f"Category keywords were not a list: {category_keywords}. Resetting.")
        category_keywords = []
    if category_label == "unknown":
         logging.info(f"LLM returned 'Unknown' for category for: '{sentence_snippet}'")
    logging.debug(f"Category result: Label='{category_label}', Keywords={category_keywords}")
    return category_label, category_keywords, "Success"


# get_llm_belbin_role remains the same (uses local if available)
def get_llm_belbin_role(sentence, category_prompt_map=BELBIN_ROLE_PROMPTS, delay=DELAY_SECONDS, max_retries=MAX_RETRIES):
    # ... (function body) ...
    sentence_snippet = sentence[:50] + '...'
    category_label, _, category_context = get_llm_category(sentence, delay=delay, max_retries=max_retries)
    if category_label in ["JSONERROR", "APIERROR", "SAFETY_BLOCKED", "LOCAL_API_ERROR"]:
        logging.error(f"Cannot get Belbin role due to error in category step ({category_label}) for: '{sentence_snippet}'")
        return category_label, [], f"Error in Category Step: {category_context}"
    category_lower = category_label.lower()
    logging.info(f"Getting Belbin role (category: {category_lower}) for: '{sentence_snippet}'")
    if category_lower == "unknown" or category_lower not in category_prompt_map:
        logging.warning(f"Category '{category_lower}' is 'Unknown' or not in prompt map.")
        return "Unknown", [], "Category Unknown or Not Mapped"
    specific_prompt = category_prompt_map[category_lower]
    if local_model_available:
        logging.debug("Using Local LLM for Belbin role prediction.")
        response_text, status_code = call_local_llm_api(specific_prompt, sentence, max_retries=max_retries)
    else:
        logging.debug("Using Gemini API for Belbin role prediction (local unavailable).")
        response_text, status_code = call_gemini_api(specific_prompt, sentence, delay=delay, max_retries=max_retries)
    error_label, result_data, error_context = _handle_api_call_result(response_text, status_code, f"role (cat: {category_lower})", sentence_snippet)
    if error_label:
        return error_label, [], error_context
    response_json = result_data
    belbin_role_label = response_json.get("belbin_role", "Unknown")
    belbin_role_keywords = response_json.get("belbin_role_keywords", [])
    belbin_role_context = response_json.get("belbin_role_context", "Unknown")
    if not isinstance(belbin_role_keywords, list):
        logging.warning(f"Belbin role keywords were not a list: {belbin_role_keywords}. Resetting.")
        belbin_role_keywords = []
    if belbin_role_label.lower() == "unknown":
         logging.info(f"LLM returned 'Unknown' for Belbin role (category {category_lower}) for: '{sentence_snippet}'")
    logging.debug(f"Belbin Role result: Label='{belbin_role_label}', Keywords={belbin_role_keywords}, Context='{belbin_role_context}'")
    return belbin_role_label, belbin_role_keywords, belbin_role_context


# get_llm_synonyms remains the same (uses local if available)
def get_llm_synonyms(sentence, delay=DELAY_SECONDS, max_retries=MAX_RETRIES):
    # ... (function body) ...
    logging.info(f"Getting LLM Belbin role via SYNONYM prompt for: '{sentence[:50]}...'")
    sentence_snippet = sentence[:50] + '...'
    if local_model_available:
        logging.debug("Using Local LLM for synonym prediction.")
        response_text, status_code = call_local_llm_api(SYNONYM_PROMPT, sentence, max_retries=max_retries)
    else:
        logging.debug("Using Gemini API for synonym prediction (local unavailable).")
        response_text, status_code = call_gemini_api(SYNONYM_PROMPT, sentence, delay=delay, max_retries=max_retries)
    error_label, result_data, error_context = _handle_api_call_result(response_text, status_code, "synonym", sentence_snippet)
    if error_label:
        return error_label, [], error_context
    response_json = result_data
    belbin_role_label = response_json.get("belbin_role", "Unknown")
    if belbin_role_label.lower() == "need more training":
         belbin_role_label = "Unknown"
         logging.info(f"LLM synonym prompt returned 'Need More Training', mapped to 'Unknown'.")
    belbin_role_keywords = response_json.get("belbin_role_keywords", [])
    belbin_role_context = response_json.get("belbin_role_context", "Unknown")
    if isinstance(belbin_role_context, str) and belbin_role_context.lower() == "need more training":
        belbin_role_context = "Unknown"
    if not isinstance(belbin_role_keywords, list):
        logging.warning(f"Synonym keywords were not a list: {belbin_role_keywords}. Resetting.")
        belbin_role_keywords = []
    if belbin_role_label.lower() == "unknown":
        logging.info(f"LLM returned 'Unknown' (or mapped equivalent) for synonym step for: '{sentence_snippet}'")
    logging.debug(f"Synonym Role result: Label='{belbin_role_label}', Keywords={belbin_role_keywords}, Context='{belbin_role_context}'")
    return belbin_role_label, belbin_role_keywords, belbin_role_context


# --- Optional/Unused Functions ---
# ... (Keep these if needed, no changes required by local LLM integration) ...