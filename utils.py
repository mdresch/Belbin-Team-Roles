# utils.py
import google.generativeai as genai
import json
import logging
import time
import os  # Import the 'os' module
from config import DELAY_SECONDS, MAX_RETRIES, GOOGLE_MODEL_NAME, FINE_TUNED_MODEL_PATH
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Initialize the *general* sentiment analysis pipeline (from transformers)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# We'll load the fine-tuned model and tokenizer here, but handle potential
# loading failures gracefully.
tokenizer = None
model = None
label_map_global = {} # Global variable to store the label map.

def load_fine_tuned_model(model_path=FINE_TUNED_MODEL_PATH):
    """Loads the fine-tuned model, tokenizer, and label_map."""
    global tokenizer, model, label_map_global  # Use global variables
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Load the label_map from the JSON file
        label_map_path = os.path.join(model_path, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, "r") as f:
                label_map_global = json.load(f)
            print(f"DEBUG: Loaded label_map: {label_map_global}")
        else:
            print(f"WARNING: label_map.json not found in {model_path}.  Belbin role/sentiment mapping will be incorrect!")
            return False # Indicate failure


        if torch.cuda.is_available():
            model = model.to('cuda')
            print("Using GPU")
        else:
            print("Using CPU")
        return True  # Indicate successful loading

    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return False  # Indicate loading failure

# Load the fine-tuned model when utils.py is imported.
model_loaded = load_fine_tuned_model()



def call_gemini_api(prompt, model_name=GOOGLE_MODEL_NAME, delay=DELAY_SECONDS, max_retries=MAX_RETRIES):
    """Calls the Gemini API with retries and error handling."""
    retries = 0
    while retries < max_retries:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as api_err:
            if hasattr(api_err, 'status_code') and api_err.status_code == 429:
                retries += 1
                delay *= 2
                logging.warning(f"Rate limit exceeded. Retrying in {delay} seconds.")
                time.sleep(delay)
            else:
                logging.error(f"Gemini API Error: {api_err}")
                return None  # Or raise the exception if you prefer
    logging.error("Max retries exceeded for API call.")
    return None


def parse_json_response(response_text):
    """Parses a JSON response from the Gemini API, handling errors."""
    try:
        json_start_index = response_text.find('{')
        json_end_index = response_text.rfind('}') + 1
        if json_start_index != -1 and json_end_index != -1:
            json_text = response_text[json_start_index:json_end_index]
            return json.loads(json_text)
        else:
            logging.warning(f"JSON start/end not found. Response: {response_text}")
            return None
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logging.error(f"Error parsing JSON. {type(e).__name__}: {e}. Response: {response_text}")
        return None

def get_llm_category(sentence, prompt, delay=DELAY_SECONDS, max_retries=MAX_RETRIES):
    """Gets the first-level category (Action, Thought, People) from the LLM."""
    response_text = call_gemini_api(prompt.replace("[SENTENCE_HERE]", sentence), delay=delay, max_retries=max_retries)
    if response_text is None:
        return "unknown", []

    response_json = parse_json_response(response_text)
    if response_json:
        category_label = response_json.get("category", "unknown").lower()
        category_keywords = response_json.get("keywords", [])
        if not isinstance(category_keywords, list):
            category_keywords = []
            logging.warning("Keywords were not a list.")
    else:
        category_label = "unknown"
        category_keywords = []

    return category_label, category_keywords


def get_llm_belbin_role(sentence, prompt, delay=DELAY_SECONDS, max_retries=MAX_RETRIES):
    """Gets the Belbin role, keywords, and context from the LLM (Gemini API)."""
    full_prompt = f"{prompt}\n\nTeams meeting contribution sentence: {sentence}"
    response_text = call_gemini_api(full_prompt, delay=delay, max_retries=max_retries)

    if response_text is None:
        return "Unknown", [], "Unknown"

    response_json = parse_json_response(response_text)
    if response_json and isinstance(response_json, dict):
        belbin_role_label = response_json.get("belbin_role", "Unknown")
        belbin_role_keywords = response_json.get("belbin_role_keywords", [])
        belbin_role_context = response_json.get("belbin_role_context", "Unknown")
        # Ensure keywords is a list
        if not isinstance(belbin_role_keywords, list):
            belbin_role_keywords = []
    else:
        belbin_role_label = "Unknown"
        belbin_role_keywords = []
        belbin_role_context = "Unknown"

    return belbin_role_label, belbin_role_keywords, belbin_role_context


def get_sentiment_with_pretrained(sentence):
    """Gets sentiment using the pre-trained Hugging Face model (general sentiment)."""

    #Using the original distibert model
    result = sentiment_pipeline(sentence)[0]  # Run the pipeline.  Result is a list of dicts.
    label = result['label']  # 'POSITIVE' or 'NEGATIVE'
    score = result['score']

    # Convert to our desired format.
    if label == 'POSITIVE':
        tone_label = "positive"
        sentiment_score = score
    elif label == 'NEGATIVE':
        tone_label = "negative"
        sentiment_score = -score  # Make it negative
    else:
        tone_label = "neutral" # Should not happen with this model, but good to have
        sentiment_score = 0.0

    # --- NEUTRAL THRESHOLDING ---
    neutral_threshold = 0.4  # Adjust this value as needed
    if -neutral_threshold < sentiment_score < neutral_threshold:
        tone_label = "neutral"
        sentiment_score = 0.0

    return {
        "tone_label": tone_label,
        "sentiment_score": sentiment_score,
        "sentiment_confidence": score,  # Use the model's confidence directly
    }

# --- NEW FUNCTION for Belbin-specific sentiment ---
def get_belbin_sentiment(sentence):
    """Gets the combined Belbin role and role-specific sentiment using the fine-tuned model."""

    if not model_loaded:
        logging.error("Fine-tuned model not loaded.  Returning default values.")
        return {
            "belbin_role": "Unknown",
            "belbin_tone": "neutral",
            "belbin_score": 0.0,
            "belbin_confidence": 0.8
        }

    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

     # Move inputs to the same device as the model
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}

    # Get model predictions
    with torch.no_grad():  # Disable gradient calculation during inference
        outputs = model(**inputs)

    # Get predicted label and confidence
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence = torch.max(probabilities, dim=-1)[0].item()  # Get confidence of predicted class

    # Map numerical label back to combined label string, adding default.
    label_map = {v: k for k, v in label_map_global.items()}

    predicted_combined_label = label_map.get(predictions[0].item(), "Unknown_neutral") # Default if not found
    belbin_role, belbin_tone = predicted_combined_label.split("_")  # Split back into role and tone


    # Convert sentiment to numerical score (optional, but good for consistency)
    if belbin_tone == "positive":
        belbin_score = confidence
    elif belbin_tone == "negative":
        belbin_score = -confidence
    else:
        belbin_score = 0.0

    return {
        "belbin_role": belbin_role,
        "belbin_tone": belbin_tone,
        "belbin_score": belbin_score,
        "belbin_confidence": confidence
    }