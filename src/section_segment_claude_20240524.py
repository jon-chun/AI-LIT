# Human-AI collaborative pair coding with Claude 3.0 Opus
# https://claude.ai/chat/866e84cb-3063-402e-a7c1-36495ecf9272

# Jon Chun
# 24 May 2024

import os # ERROR #2: forgot this line to import os
import re
import logging
from typing import List, Tuple  # ERROR #1: forgot to add ", Tuple"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

MIN_CHARS_PARAGRAPH = 50
MIN_WORDS_PARAGRAPH = 5

def create_base_subdir(model_config: str) -> Tuple[str, str, str, str]:  # ERROR #1: see "import typing..." above
    """
    Create base subdirectories for a given model configuration.

    Args:
    - model_config (str): The model configuration string.

    Returns:
    - tuple: Four base directory paths (truth, sample, score, input).
    """
    try:
        # Define a lookup for subdir_in based on the model configuration
        subdir_lookup = {
            "llama3-8b-q4km": os.path.join("..", "data", "wikifilmsum_large_filtered"),
            "phi3-38b-q4km": os.path.join("..", "data", "wikifilmsum_small_filtered"),
            "mistral-7b-instr-q4km": os.path.join("..", "data", "wikifilmsum_small_filtered"),
        }

        # Base directory paths
        base_truth_dir = os.path.join("..", "data", "wikifilmsum_small_filtered")
        base_sample_dir = os.path.join("..", "data", "step1out_generate_narrative_elements", model_config)
        base_score_dir = os.path.join("..", "data", "step2out_score_narrative_elements", model_config)
        
        # Dynamic input directory based on the model configuration
        subdir_in = subdir_lookup.get(model_config, os.path.join("..", "data", "default_directory"))
        
        # Ensure base directories exist
        ensure_dir_exists(base_truth_dir)
        ensure_dir_exists(base_sample_dir)
        ensure_dir_exists(base_score_dir)
        ensure_dir_exists(subdir_in)

        return base_truth_dir, base_sample_dir, base_score_dir, subdir_in
    except Exception as e:
        logging.error(f"Error creating base subdirectories: {e}")
        raise
    
def is_title_case(s: str) -> bool:
    """Check if a string is in title case."""
    return s == s.title()

def ensure_dir_exists(path: str) -> None:
    """Ensure that the directory exists; if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)

def clean_paragraphs(paragraphs_raw_list: List[str]) -> List[str]:
    """
    Clean and filter a list of raw paragraphs based on specific criteria.

    Args:
    - paragraphs_raw_list (List[str]): List of raw paragraphs to be cleaned.

    Returns:
    - List[str]: List of cleaned paragraphs.
    """
    if not paragraphs_raw_list:
        return []

    pattern = re.compile(r'^(CHAPTER|SECTION|PART|Chapter|Section|Part)[\s.,;:!?\-«»]{0,5}(\d+|[IVXLCDM]+|[a-zA-Z]+)\b')
    paragraphs_clean_list = []

    try:
        for paragraph_raw_now in paragraphs_raw_list:
            logging.debug(f"Processing paragraph: {paragraph_raw_now}")
            paragraph_clean_now = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', paragraph_raw_now)

            # Check for alphabet letters (including French characters)
            if not re.search(r'[a-zA-ZÀ-ÖØ-öø-ÿ]', paragraph_clean_now):
                logging.debug("Skipping paragraph with no alphabet letters.")
                continue

            # Normalize whitespace and calculate length
            normalized_paragraph = re.sub(r'\s+', ' ', paragraph_clean_now).strip()
            if len(normalized_paragraph) <= MIN_CHARS_PARAGRAPH:
                logging.debug("Skipping paragraph with insufficient length.")
                continue

            # Check for all caps and word count
            words = normalized_paragraph.split()
            all_caps_word_count = sum(1 for word in words if word.isupper())
            if all_caps_word_count > MIN_WORDS_PARAGRAPH:
                logging.debug("Skipping paragraph with too many all caps words.")
                continue

            # Check for the specific pattern at the start
            if pattern.match(normalized_paragraph):
                # Extract the potential subtitle/chapter heading
                remaining_text = pattern.sub('', normalized_paragraph).strip()
                remaining_words = remaining_text.split()
                
                # Check if the remaining words form a subtitle/chapter heading
                if len(remaining_words) <= 7 and (all(word.isupper() for word in remaining_words) or is_title_case(remaining_text)):
                    logging.debug("Skipping paragraph with chapter/section heading.")
                    continue

            # If all checks pass, add to the clean list
            paragraphs_clean_list.append(paragraph_clean_now)
            logging.debug("Paragraph added to clean list.")
    
    except Exception as e:
        logging.error(f"Error processing paragraphs: {e}")

    return paragraphs_clean_list

if __name__ == "__main__":
    model_config = "phi3-38b-q4km"
    base_truth_dir, base_sample_dir, base_score_dir, subdir_in = create_base_subdir(model_config)

    # Replace this with your actual list of raw paragraphs
    paragraphs_raw_list = [
        "The Way by Swann's",
        "For Monsieur Gaston Calmette",
        "As a token of profound",
        "and affectionate gratitude.",
        "Marcel Proust",
        "PART I: Combray",
        "1",
        # Add more paragraphs as needed
    ]

    # Process the paragraphs
    paragraphs_clean_list = clean_paragraphs(paragraphs_raw_list)

    # Print cleaned paragraphs for verification
    for paragraph in paragraphs_clean_list:
        print(paragraph)