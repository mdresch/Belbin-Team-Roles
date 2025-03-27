# main.py
import time
import pandas as pd
import google.generativeai as genai
import logging # Keep logging import early for potential use in error handling
import datetime
import io
import os
import traceback
import json
import sys

# --- Environment/Path Debugging (Helps find modules) ---
print("--- Debugging Import Paths ---")
print(f"Current Working Directory: {os.getcwd()}")
print("Python sys.path:")
# Use a try-except block for __file__ in case it's not defined (e.g., interactive session)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_sys_path = script_dir in sys.path
    print(f"Script Directory: {script_dir} (Exists: {os.path.isdir(script_dir)}, In sys.path: {in_sys_path})")
    if not in_sys_path:
        print(f"WARNING: Script directory '{script_dir}' not found in sys.path. Adding it.")
        sys.path.insert(0, script_dir) # Add script's directory to the beginning
except NameError:
    script_dir = os.getcwd() # Fallback if __file__ is not defined
    print(f"Warning: __file__ not defined. Using current working directory as script directory: {script_dir}")
    if script_dir not in sys.path:
         print(f"WARNING: CWD '{script_dir}' not found in sys.path. Adding it.")
         sys.path.insert(0, script_dir)

print("Current Python sys.path:")
for path_entry in sys.path:
    print(f"  - {path_entry}")
print("--- End Debugging Import Paths ---")


# --- Configuration Import ---
print("Attempting to import from config...")
try:
    # Import necessary config variables individually
    from config import GOOGLE_API_KEY, OUTPUT_CSV_PATH, LOG_LEVEL, DELAY_SECONDS, MAX_RETRIES
    print(f"Successfully imported config variables. DELAY_SECONDS type: {type(DELAY_SECONDS)}, value: {DELAY_SECONDS}")
except ImportError as e:
    print(f"FATAL ERROR: Cannot import required variables from config.py: {e}")
    print("Check if config.py exists in the script directory/sys.path and contains all required variables.")
    raise # Halt execution
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred importing config.py: {e}")
    raise # Halt execution


# --- Functionality Imports ---
print("Attempting to import from utils...")
try:
    # Import necessary utils functions individually
    from utils import get_llm_category, get_llm_belbin_role, get_llm_synonyms
    print("Successfully imported functions from utils.")
except ImportError as e:
    print(f"FATAL ERROR: Failed to import functions from utils.py: {e}")
    print("Check if utils.py exists and contains the required function definitions.")
    raise # Halt execution
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred importing utils.py: {e}")
    raise # Halt execution


# --- Setup logging ---
# This section now runs only AFTER config import was successful
print("Setting up logging...")
log_buffer = io.StringIO()
# Validate LOG_LEVEL from config
log_level_name = LOG_LEVEL.upper()
if not hasattr(logging, log_level_name):
    print(f"Warning: Invalid LOG_LEVEL '{LOG_LEVEL}' in config.py. Defaulting to INFO.")
    log_level_name = "INFO"
log_level_config = getattr(logging, log_level_name)

# Define formatter and handlers
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
buffer_handler = logging.StreamHandler(log_buffer)
buffer_handler.setFormatter(log_formatter)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(log_formatter)

# Configure root logger
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear() # Clear handlers if logger was configured previously
root_logger.setLevel(log_level_config)
root_logger.addHandler(buffer_handler)
root_logger.addHandler(stdout_handler)
print("Logging setup complete.")


# --- Analysis Function Definition ---
# This line is now reached only if config import succeeded and DELAY_SECONDS/MAX_RETRIES exist
print("Defining analyze_contribution function...")
def analyze_contribution(sentence, delay=DELAY_SECONDS, max_retries=MAX_RETRIES):
    """
    Performs Belbin role analysis, distinguishing LLM Unknown from parsing/API/safety errors.

    Args:
        sentence (str): The sentence to analyze.
        delay (int): Base delay between API calls.
        max_retries (int): Max retries for API calls.

    Returns:
        dict: Contains analysis results including predicted category and role,
              keywords, context, and potential error states.
    """
    # Use logger specific to this function for better context
    logger = logging.getLogger(__name__ + ".analyze_contribution")
    logger.debug(f"Analyzing sentence: '{sentence}'")
    sentence_snippet = sentence[:50] + '...' # For concise logging

    # Initialize default/error states
    initial_role_label, initial_role_keywords, initial_role_context = "Unknown", [], "Unknown"
    final_role_label, final_role_keywords, final_role_context = "Unknown", [], "Unknown"
    predicted_category, predicted_category_keywords = "Unknown", []

    try:
        # --- Step 1 & 2: Get Category and Initial Belbin Role ---
        # get_llm_belbin_role handles category step internally and propagates errors.
        # Possible return labels: Valid Role, "Unknown", "JSONERROR", "APIERROR", "SAFETY_BLOCKED"
        logger.info(f"Getting initial Belbin role for: '{sentence_snippet}'")
        initial_role_label, initial_role_keywords, initial_role_context = get_llm_belbin_role(
            sentence, delay=delay, max_retries=max_retries
        )
        logger.info(f"Initial Role Call Result: Role='{initial_role_label}', Context='{initial_role_context}'")

        # --- Step 3: Fallback ONLY if initial role is specifically "Unknown" (LLM ambiguity) ---
        if initial_role_label.lower() == "unknown":
            logger.warning(f"Initial role was 'Unknown'. Trying fallback for: '{sentence_snippet}'")
            try:
                # Call the synonym/fallback function
                # Possible return labels: Valid Role, "Unknown", "JSONERROR", "APIERROR", "SAFETY_BLOCKED"
                fb_role_label, fb_role_keywords, fb_role_context = get_llm_synonyms(
                    sentence, delay=delay, max_retries=max_retries
                )
                logger.info(f"Fallback Role Call Result: Role='{fb_role_label}', Context='{fb_role_context}'")

                # Decide whether to use fallback result:
                if fb_role_label.lower() not in ["jsonerror", "apierror", "safety_blocked"]:
                    # Fallback succeeded OR was 'Unknown' itself - use its result
                    final_role_label, final_role_keywords, final_role_context = fb_role_label, fb_role_keywords, fb_role_context
                    if fb_role_label.lower() == "unknown":
                         logger.warning(f"Fallback also resulted in 'Unknown' for: '{sentence_snippet}'")
                else:
                    # Fallback resulted in an error - discard fallback, keep initial 'Unknown'
                    logger.error(f"Fallback attempt errored ({fb_role_label}). Keeping 'Unknown' from initial attempt for: '{sentence_snippet}'")
                    final_role_label, final_role_keywords, final_role_context = initial_role_label, initial_role_keywords, initial_role_context

            except Exception as fallback_err:
                 logger.error(f"Exception during fallback analysis for '{sentence_snippet}': {fallback_err}", exc_info=True)
                 # Fallback itself crashed - keep initial 'Unknown'
                 final_role_label, final_role_keywords, final_role_context = initial_role_label, initial_role_keywords, initial_role_context
        else:
             # Initial role was successful OR had a JSON/API/Safety error. Use that result directly.
             final_role_label = initial_role_label
             final_role_keywords = initial_role_keywords
             final_role_context = initial_role_context

    except Exception as e:
        # Catch unexpected errors during the main analysis flow
        logger.error(f"Unhandled Exception during primary analysis for '{sentence_snippet}': {e}", exc_info=True)
        final_role_label = "PROCESSERROR" # Use distinct state
        final_role_keywords = []
        final_role_context = f"Unhandled Exception: {e}"

    # --- Get Category Separately (for reporting consistency) ---
    # Ensures category info is available even if role step failed early.
    # Handles category-specific errors (JSON/API/Safety).
    try:
        # Possible return labels: Valid Category, "Unknown", "JSONERROR", "APIERROR", "SAFETY_BLOCKED"
        logger.info(f"Getting category separately for reporting: '{sentence_snippet}'")
        # Pass context using '_' as it's not used further here
        cat_label, cat_keywords, _ = get_llm_category(sentence, delay=delay, max_retries=max_retries)
        predicted_category = cat_label
        predicted_category_keywords = cat_keywords

        # If role analysis failed due to a category error, ensure predicted_category reflects that.
        # Check if final_role_context is a string before checking content
        role_context_str = str(final_role_context)
        if final_role_label in ["JSONERROR", "APIERROR", "SAFETY_BLOCKED"] and "Category Step" in role_context_str:
             if predicted_category != final_role_label:
                  logger.warning(f"Aligning predicted_category '{predicted_category}' with role error state '{final_role_label}' due to context '{role_context_str}'.")
                  predicted_category = final_role_label # Align category state

    except Exception as cat_err:
         logger.error(f"Exception getting category separately for '{sentence_snippet}': {cat_err}", exc_info=True)
         predicted_category = "PROCESSERROR" # Use distinct state
         predicted_category_keywords = []


    return {
        "sentence": sentence,
        "predicted_category": predicted_category,
        "predicted_category_keywords": predicted_category_keywords,
        "predicted_belbin_role": final_role_label,
        "predicted_belbin_role_keywords": final_role_keywords,
        "predicted_belbin_role_context": final_role_context,
    }

# --- Findings Helper Definition ---
print("Defining get_findings function...")
def get_findings(predicted_category, predicted_belbin_role):
    """Determines findings, distinguishing Unknown from technical/safety errors."""
    logger = logging.getLogger(__name__ + ".get_findings") # Use specific logger
    category_str = str(predicted_category)
    role_str = str(predicted_belbin_role)
    category_lower = category_str.lower()
    role_lower = role_str.lower()

    # Define all possible error states (lowercase)
    error_states = ["jsonerror", "apierror", "processerror", "safety_blocked", "fatalerror"]

    # Check for technical/safety errors first
    category_is_error = category_lower in error_states
    role_is_error = role_lower in error_states

    if category_is_error or role_is_error:
        # Determine primary error source (role error takes precedence if both exist)
        error_source_label = role_str if role_is_error else category_str
        logger.warning(f"Reporting error finding: Category='{category_str}', Role='{role_str}', Source='{error_source_label}'")
        if "safety" in error_source_label.lower():
             return f"Safety Block during analysis ({error_source_label})"
        else:
             return f"Technical Error during analysis ({error_source_label})"
    # Check for LLM ambiguity (only if no technical errors)
    elif category_lower == 'unknown' or role_lower == 'unknown':
        logger.info(f"Reporting ambiguous finding: Category='{category_str}', Role='{role_str}'")
        return "Analysis ambiguous (LLM returned Unknown)"
    # If no errors and not Unknown, analysis is complete
    else:
        logger.debug(f"Reporting complete finding: Category='{category_str}', Role='{role_str}'")
        return f"Analysis complete: Category={category_str}, Belbin Role={role_str}"

# --- Main Execution Definition ---
print("Defining main function...")
def main():
    """Main function to load data, run analysis, and save results."""
    logger = logging.getLogger(__name__ + ".main") # Use specific logger
    script_start_time = time.time()
    logger.info(f"Starting main execution at {datetime.datetime.now(datetime.timezone.utc).isoformat()}")

    # --- Configure Gemini ---
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Google Generative AI SDK configured.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to configure Google Generative AI SDK: {e}", exc_info=True)
        # Use print as logger might not reach file/console if exit happens
        print(f"FATAL: Failed to configure Google Generative AI SDK: {e}")
        sys.exit(1) # Force exit

    # --- Load test sentences ---
    sentences_path = 'sentences.csv' # Assumed relative path
    try:
        logger.info(f"Attempting to load sentences from: {os.path.abspath(sentences_path)}")
        test_sentences_df = pd.read_csv(sentences_path)
        logger.info(f"Successfully loaded {len(test_sentences_df)} rows from {sentences_path}")

        # Validate required columns
        required_cols = ['sentence', 'belbin_category', 'expected_role']
        missing_cols = [col for col in required_cols if col not in test_sentences_df.columns]
        if missing_cols:
            logger.critical(f"Missing required columns in {sentences_path}: {missing_cols}")
            print(f"ERROR: Missing required columns in {sentences_path}: {missing_cols}")
            sys.exit(1) # Force exit

        # Preprocess data
        test_sentences_df['belbin_category'] = test_sentences_df['belbin_category'].str.lower().fillna('unknown')
        test_sentences_df['expected_role'] = test_sentences_df['expected_role'].fillna('Unknown')
        # Ensure sentence is string and strip leading/trailing whitespace
        test_sentences_df['sentence'] = test_sentences_df['sentence'].astype(str).str.strip()

        test_sentences_expected = test_sentences_df.to_dict('records')
        logger.info(f"Prepared {len(test_sentences_expected)} test cases for processing.")

    except FileNotFoundError:
        logger.critical(f"ERROR: sentences.csv not found at {os.path.abspath(sentences_path)}.")
        print(f"ERROR: sentences.csv not found at {os.path.abspath(sentences_path)}.")
        sys.exit(1) # Force exit
    except Exception as e:
        # Catch pandas parsing errors or other file issues
        logger.critical(f"ERROR loading or processing sentences.csv: {e}", exc_info=True)
        print(f"ERROR loading or processing sentences.csv: {e}")
        sys.exit(1) # Force exit

    # --- Analysis Loop ---
    analysis_results = []
    csv_headers = [ # Ensure order and completeness
        "sentence",
        "predicted_category", "expected_category", "category_match",
        "predicted_belbin_role", "expected_belbin_role", "role_match",
        "predicted_category_keywords",
        "predicted_belbin_role_keywords", "predicted_belbin_role_context",
        "findings", "timestamp"
    ]

    print("\n--- Starting Analysis Loop ---") # User feedback
    total_cases = len(test_sentences_expected)
    logger.info(f"Starting analysis loop for {total_cases} sentences...")

    for i, test_case in enumerate(test_sentences_expected):
        sentence = test_case.get("sentence", "") # Default to empty string if missing
        expected_category = test_case.get("belbin_category") # Already lowercased/filled
        expected_role = test_case.get("expected_role") # Already filled

        # Skip empty sentences after stripping whitespace
        if not sentence:
            logger.warning(f"Skipping row {i+1}/{total_cases} due to empty sentence.")
            continue

        # User feedback (print)
        print(f"\n[{i+1}/{total_cases}] Processing: \"{sentence[:80]}...\"")
        # Detailed log
        logger.info(f"Processing test case {i+1}/{total_cases}. Sentence: '{sentence}'")

        # Initialize result dict for this row
        row_result = {hdr: pd.NA for hdr in csv_headers} # Use pd.NA for better CSV/DB handling
        row_result["sentence"] = sentence
        row_result["expected_category"] = expected_category
        row_result["expected_belbin_role"] = expected_role
        row_result["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat() # Use UTC timestamp

        try:
            # --- Call the core analysis function ---
            analysis_start_time = time.time()
            predicted_results = analyze_contribution(sentence)
            analysis_duration = time.time() - analysis_start_time
            logger.debug(f"Analysis for sentence {i+1} took {analysis_duration:.2f}s.")

            # Store predictions (can include ERROR/SAFETY states)
            row_result["predicted_category"] = predicted_results.get('predicted_category', 'Unknown')
            # Store keywords as JSON strings - ensure they are lists first
            pred_cat_kw = predicted_results.get('predicted_category_keywords', [])
            row_result["predicted_category_keywords"] = json.dumps(pred_cat_kw if isinstance(pred_cat_kw, list) else [])
            row_result["predicted_belbin_role"] = predicted_results.get('predicted_belbin_role', 'Unknown')
            pred_role_kw = predicted_results.get('predicted_belbin_role_keywords', [])
            row_result["predicted_belbin_role_keywords"] = json.dumps(pred_role_kw if isinstance(pred_role_kw, list) else [])
            row_result["predicted_belbin_role_context"] = predicted_results.get('predicted_belbin_role_context', 'Unknown')

            # Perform Comparisons (case-insensitive, handle potential ERROR/SAFETY states)
            pred_cat_str = str(row_result["predicted_category"]).lower()
            pred_role_str = str(row_result["predicted_belbin_role"]).lower()
            exp_cat_str = str(expected_category).lower()
            exp_role_str = str(expected_role).lower()
            error_states_lower = ["jsonerror", "apierror", "processerror", "safety_blocked", "fatalerror"]

            # Match only if prediction is not an error and matches expectation
            row_result["category_match"] = "✅" if pred_cat_str not in error_states_lower and pred_cat_str == exp_cat_str else "❌"
            row_result["role_match"] = "✅" if pred_role_str not in error_states_lower and pred_role_str == exp_role_str else "❌"

            # Get findings using the updated function
            row_result["findings"] = get_findings(row_result["predicted_category"], row_result["predicted_belbin_role"])

            # Print summary for user
            print(f"  Category:    Expected: {expected_category:<20} | Predicted: {row_result['predicted_category']:<20} {row_result['category_match']}")
            print(f"  Belbin Role: Expected: {expected_role:<20} | Predicted: {row_result['predicted_belbin_role']:<20} {row_result['role_match']}")
            print(f"  Findings:      {row_result.get('findings', 'N/A')}")
            logger.info(f"Result for sentence {i+1}: Cat={row_result['predicted_category']}({row_result['category_match']}), Role={row_result['predicted_belbin_role']}({row_result['role_match']}), Findings={row_result['findings']}")

        except Exception as e:
            # Catch unexpected fatal errors during the loop processing itself
            logger.critical(f"FATAL ERROR processing sentence id {i+1} ('{sentence[:50]}...'): {e}", exc_info=True)
            print(f"FATAL ERROR processing sentence id {i+1}: {e}")
            # Record error state
            row_result["predicted_category"] = "FATALERROR"
            row_result["predicted_belbin_role"] = "FATALERROR"
            row_result["category_match"] = "❌"
            row_result["role_match"] = "❌"
            row_result["findings"] = f"Fatal Processing Error: {e}"
            row_result["predicted_category_keywords"] = json.dumps([])
            row_result["predicted_belbin_role_keywords"] = json.dumps([])
            row_result["predicted_belbin_role_context"] = f"Fatal Error: {e}"

        analysis_results.append(row_result)
        # time.sleep(0.1) # Optional small delay if rate limits are aggressive

    print("\n--- Analysis Loop Finished ---") # User feedback
    logger.info(f"Finished analysis loop for {total_cases} sentences.")

    # --- Save results to CSV ---
    if not analysis_results:
        logger.warning("No analysis results generated to save.")
        print("\nNo analysis results generated to save.")
    else:
        # Convert results list to DataFrame, ensuring columns exist and are ordered
        df_results = pd.DataFrame(analysis_results, columns=csv_headers)

        logger.info(f"Attempting to save {len(df_results)} results to {OUTPUT_CSV_PATH}")
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(OUTPUT_CSV_PATH)
            # Handle case where path is just a filename (no directory part)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")

            file_exists = os.path.exists(OUTPUT_CSV_PATH)

            # Append or create CSV using robust mode/header logic
            df_results.to_csv(
                OUTPUT_CSV_PATH,
                mode='a',           # Append if file exists
                header=not file_exists, # Write header only if creating new file
                index=False,
                columns=csv_headers # Ensure column order/selection
            )

            action = "Appended" if file_exists else "Created"
            logger.info(f"Successfully {action} {len(df_results)} records to {OUTPUT_CSV_PATH}")
            print(f"\n{action} {len(df_results)} results to {OUTPUT_CSV_PATH}")

        except Exception as e:
            logger.error(f"Error saving CSV to {OUTPUT_CSV_PATH}: {e}", exc_info=True)
            print(f"Error saving CSV to {OUTPUT_CSV_PATH}: {e}")

    # --- Final Log Output & Summary ---
    script_end_time = time.time()
    duration = script_end_time - script_start_time
    logger.info(f"Script finished processing {len(analysis_results)} records in {duration:.2f} seconds.")
    print(f"\nScript finished in {duration:.2f} seconds.")

    # Save log buffer to file
    # Handle case where OUTPUT_CSV_PATH might be just a filename
    log_dir = os.path.dirname(OUTPUT_CSV_PATH) or '.'
    log_file_path = os.path.join(log_dir, 'analysis_log.txt')
    try:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
             # Add separator and timestamp for clarity between runs
             log_file.write(f"\n--- Log Session End: {datetime.datetime.now(datetime.timezone.utc).isoformat()} ---\n")
             log_file.write("--- Log Buffer Content For This Run: ---\n")
             log_file.write(log_buffer.getvalue())
             log_file.write(f"--- End Log Buffer (Script Duration: {duration:.2f}s) ---\n\n")
        print(f"Full log appended to {log_file_path}")
    except Exception as e:
        print(f"\nError saving full log to {log_file_path}: {e}")


# --- Script Entry Point ---
print("Checking if running as main script...")
if __name__ == "__main__":
    print("Running main execution block...")
    # Ensure handlers are cleared if run multiple times in same environment
    # Useful for notebooks or frameworks that might keep logger state
    logging.getLogger().handlers.clear()
    main() # Call the main function
    print("Main execution finished.")