# main.py
import time
import pandas as pd
import google.generativeai as genai
import logging
import datetime
import io
import os
import traceback
# Use relative imports:
from config import GOOGLE_API_KEY, OUTPUT_CSV_PATH, LOG_LEVEL, DELAY_SECONDS, MAX_RETRIES
from prompts import FIRST_LEVEL_PROMPT, BELBIN_ROLE_PROMPTS  # Removed TONE_PROMPTS
from utils import get_llm_category, get_llm_belbin_role, get_sentiment_with_pretrained, get_belbin_sentiment


# --- Setup logging ---
log_buffer = io.StringIO()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),  # Dynamic level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(log_buffer)]
)


def analyze_contribution(sentence, delay=DELAY_SECONDS, max_retries=MAX_RETRIES):
    """Performs three-iteration Belbin role analysis on a sentence."""

    # --- Iteration 1: Category (Action, Thought, People) ---
    first_level_category, category_keywords = get_llm_category(sentence, FIRST_LEVEL_PROMPT, delay, max_retries)

    # --- Iteration 2: Belbin Role, Keywords, and Context (Gemini API) ---
    belbin_role_label, belbin_role_keywords, belbin_role_context = "Unknown", [], "Unknown"  # Defaults
    if first_level_category.lower() in BELBIN_ROLE_PROMPTS:
        belbin_role_prompt = BELBIN_ROLE_PROMPTS[first_level_category.lower()]
        belbin_role_label, belbin_role_keywords, belbin_role_context = get_llm_belbin_role(sentence, belbin_role_prompt, delay, max_retries)


    # --- Iteration 3: General Sentiment (Pre-trained DistilBERT) ---
    general_sentiment = get_sentiment_with_pretrained(sentence)

    # --- Iteration 4: Belbin-Specific Sentiment (Fine-Tuned Model) ---
    belbin_sentiment = get_belbin_sentiment(sentence)


    return {
        "sentence": sentence,
        "predicted_category": first_level_category,
        "predicted_category_keywords": category_keywords,
        "predicted_belbin_role": belbin_role_label,          # From Gemini API
        "predicted_belbin_role_keywords": belbin_role_keywords,  # From Gemini API
        "predicted_belbin_role_context": belbin_role_context,  # From Gemini API
        "predicted_belbin_tone": belbin_sentiment["belbin_tone"],  # From fine-tuned model
        "predicted_belbin_score": belbin_sentiment["belbin_score"], # From fine-tuned model
        "predicted_belbin_confidence": belbin_sentiment["belbin_confidence"], # From fine-tuned model
        "predicted_tone": general_sentiment["tone_label"],       # General sentiment
        "predicted_sentiment_score": general_sentiment["sentiment_score"], # General sent.
        "predicted_sentiment_confidence": general_sentiment.get("sentiment_confidence", 0.8), # General sent. confidence, added default.
    }

def main():
    """Main function to run the analysis."""
    genai.configure(api_key=GOOGLE_API_KEY)

    test_sentences_expected = [ #Keep expected general tone for comparison.
        {"sentence": "We need to ensure we stick to the agenda and achieve our objectives for this meeting.",
         "expected_category": "action", "expected_role": "Shaper", "expected_tone": "Neutral"},
        {"sentence": "Has anyone considered the potential risks associated with this plan?",
         "expected_category": "thought", "expected_role": "Monitor Evaluator", "expected_tone": "Neutral"},
        {"sentence": "I'm happy to help facilitate and ensure everyone contributes.",
         "expected_category": "people", "expected_role": "Co-ordinator", "expected_tone": "Positive"},
        {"sentence": "This approach might not be the most cost-effective solution; let's explore alternatives.",
         "expected_category": "thought", "expected_role": "Monitor Evaluator", "expected_tone": "Neutral"},
        {"sentence": "I'll take the lead on organizing the follow-up actions from this meeting.",
         "expected_category": "action", "expected_role": "Implementer", "expected_tone": "Neutral"},
        {"sentence": "I have some data from a recent study that could be relevant to this discussion.",
         "expected_category": "thought", "expected_role": "Specialist", "expected_tone": "Neutral"},
        {"sentence": "I think we're making good progress, and the team is working well together.",
         "expected_category": "people", "expected_role": "Teamworker", "expected_tone": "Positive"},
        {"sentence": "We need to be more innovative in our approach. Let's think outside the box.",
         "expected_category": "thought", "expected_role": "Plant", "expected_tone": "Positive"},
        {"sentence": "Let's aim to have all deliverables completed by Friday.",
         "expected_category": "action", "expected_role": "Completer Finisher", "expected_tone": "Neutral"}
    ]

    analysis_results = []
     # Updated headers to include both general and Belbin-specific sentiment
    csv_headers = ["sentence", "predicted_category", "expected_category",
                   "predicted_belbin_role", "expected_belbin_role", "predicted_belbin_role_keywords", "predicted_belbin_role_context",
                   "predicted_belbin_tone", "predicted_belbin_score", "predicted_belbin_confidence",
                   "predicted_tone", "expected_tone", "predicted_sentiment_score", "predicted_sentiment_confidence",
                   "timestamp", "predicted_category_keywords"]


    print("\n--- Test Results ---")

    for test_case in test_sentences_expected:
        sentence = test_case["sentence"]
        expected_category = test_case["expected_category"]
        expected_role = test_case["expected_role"]
        expected_tone = test_case["expected_tone"]  # This is still the *general* expected tone

        predicted_results = analyze_contribution(sentence)

        category_match = "✅" if predicted_results['predicted_category'] == expected_category else "❌"
        role_match = "✅" if predicted_results['predicted_belbin_role'] == expected_role else "❌"
        # Compare against the *general* expected tone:
        tone_match = "✅" if predicted_results['predicted_tone'].lower() == expected_tone.lower() else "❌"

        print(f"\nSentence: \"{sentence}\"")
        print(f"  Category: Expected: {expected_category}  |  Predicted: {predicted_results['predicted_category']}  {category_match}")
        print(f"  Keywords: Category: {predicted_results['predicted_category_keywords']}")
        print(f"  Belbin Role: Expected: {expected_role} |  Predicted: {predicted_results['predicted_belbin_role']}  {role_match}")
        print(f"  Belbin Role Keywords: {predicted_results['predicted_belbin_role_keywords']}")
        print(f"  Belbin Role Context: {predicted_results['predicted_belbin_role_context']}")

        # Now print *both* general and Belbin-specific sentiment:
        print(f"  General Tone: Expected: {expected_tone}     |  Predicted: {predicted_results['predicted_tone']}, Sentiment_Score: {predicted_results['predicted_sentiment_score']:.4f}, Sentiment_Confidence: {predicted_results['predicted_sentiment_confidence']:.4f}  {tone_match}")
        print(f"  Belbin Tone: Predicted: {predicted_results['predicted_belbin_tone']}, Belbin_Score: {predicted_results['predicted_belbin_score']:.4f}, Belbin_Confidence: {predicted_results['predicted_belbin_confidence']:.4f}")

        analysis_results.append({
            "sentence": sentence,
            "predicted_category": predicted_results['predicted_category'],
            "expected_category": expected_category,
            "timestamp": datetime.datetime.now(),
            "predicted_category_keywords": predicted_results['predicted_category_keywords'],
            "predicted_belbin_role": predicted_results['predicted_belbin_role'],
            "expected_belbin_role": expected_role,
            "predicted_belbin_role_keywords": predicted_results['predicted_belbin_role_keywords'],
            "predicted_belbin_role_context": predicted_results['predicted_belbin_role_context'],
            "predicted_belbin_tone": predicted_results["predicted_belbin_tone"],  # Belbin tone
            "predicted_belbin_score": predicted_results["predicted_belbin_score"],  # Belbin score
            "predicted_belbin_confidence": predicted_results["predicted_belbin_confidence"],  # Belbin confidence
            "predicted_tone": predicted_results['predicted_tone'],  # General tone
            "expected_tone": expected_tone,  # Keep the expected general tone
            "predicted_sentiment_score": predicted_results['predicted_sentiment_score'],  # General score
            "predicted_sentiment_confidence": predicted_results['predicted_sentiment_confidence'], # General conf

        })

    df_results = pd.DataFrame(analysis_results)
    try:
        if os.path.exists(OUTPUT_CSV_PATH):
            existing_df = pd.read_csv(OUTPUT_CSV_PATH)
            combined_df = pd.concat([existing_df, df_results], ignore_index=True)
            combined_df.to_csv(OUTPUT_CSV_PATH, index=False, columns=csv_headers)
            logging.info(f"Appended to {OUTPUT_CSV_PATH}")
        else:
            df_results.to_csv(OUTPUT_CSV_PATH, index=False, columns=csv_headers)
            logging.info(f"Created {OUTPUT_CSV_PATH}")
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")
        logging.error(traceback.format_exc())

    finally:
        log_output = log_buffer.getvalue()
        if log_output.strip():
            print("\n-----DEBUG LOGS-----")
            print(log_output)
        else:
            print("\nNo debug messages logged.")


if __name__ == "__main__":
    main()