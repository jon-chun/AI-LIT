# Human-AI collaborative pair coding with GPT4 w/Python Extension
# https://chatgpt.com/share/9c0d73ae-70b3-4769-a994-95a613b51945

# Jon Chun
# 24 May 2024


import os
import re
import logging
from typing import List, Tuple  # ERROR #1: was "Tuple"

MIN_CHARS_PARAGRAPH = 50   # ERROR #2: missing definition
MIN_WORDS_PARAGRAPH = 5    # ERROR #3: missing definition

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir_exists(path: str) -> None:
    """Ensure that the directory exists; if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)

def count_words(paragraph: str) -> int:
    """Count words in a paragraph."""
    return len(paragraph.split())

def is_title_case(s: str) -> bool:
    """Check if a string is in title case."""
    return s == s.title()

def has_alphabet_letters(text: str) -> bool:
    """Check if the text contains alphabet letters (including French characters)."""
    return bool(re.search(r'[a-zA-ZÀ-ÿ]', text))

def count_chars(text: str) -> int:
    """Count the number of characters, normalizing whitespace and removing non-alphanumeric characters."""
    return len(re.sub(r'\s+', ' ', re.sub(r'[^a-zA-ZÀ-ÿ0-9\s«»]', '', text)))

def count_all_caps_words(text: str) -> int:
    """Count the number of all caps words in the text."""
    return len(re.findall(r'\b[A-ZÀ-Ý]+\b', text))

def is_chapter_heading(text: str) -> bool:
    """Check if the text is a chapter heading."""
    pattern = r'^(?:CHAPTER|SECTION|PART|Chapter|Section|Part)(?:\W{0,5}\w+){0,2}(?:\W{0,5}(?:[A-ZÀ-Ý][a-zà-ÿ]+\s*){0,7})?$'
    return bool(re.match(pattern, text, re.IGNORECASE))

def create_base_subdir(model_config: str) -> Tuple[str, str, str, str]:
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

def clean_paragraphs(paragraphs_raw_list: List[str], min_chars_paragraph: int = 50, min_words_paragraph: int = 5) -> List[str]:
    """
    Clean and filter a list of raw paragraphs based on specific criteria.

    Args:
    - paragraphs_raw_list (List[str]): List of raw paragraphs to be cleaned.
    - min_chars_paragraph (int): Minimum number of characters for a paragraph.
    - min_words_paragraph (int): Minimum number of words in all caps for a paragraph.

    Returns:
    - List[str]: List of cleaned paragraphs.
    """
    pattern = re.compile(r'^(CHAPTER|SECTION|PART|Chapter|Section|Part)[\s\.,;:!?\-«»]{0,5}(\d+|[IVXLCDM]+|[a-zA-Z]+)\b')
    paragraphs_clean_list = []

    try:
        for paragraph_raw_now in paragraphs_raw_list[:50]:
            logging.debug(f"Processing paragraph: {paragraph_raw_now}")
            paragraph_clean_now = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', paragraph_raw_now)

            # Filter out paragraphs based on the rules
            if not has_alphabet_letters(paragraph_clean_now):
                logging.debug("Skipping paragraph with no alphabet letters.")
                continue

            if count_chars(paragraph_clean_now) <= min_chars_paragraph:
                logging.debug("Skipping paragraph with insufficient length.")
                continue

            if count_all_caps_words(paragraph_clean_now) <= min_words_paragraph:
                logging.debug("Skipping paragraph with too many all caps words.")
                continue

            if is_chapter_heading(paragraph_clean_now):
                logging.debug("Skipping paragraph with chapter/section heading.")
                continue

            paragraphs_clean_list.append(paragraph_clean_now)
            logging.debug("Paragraph added to clean list.")
    
    except Exception as e:
        logging.error(f"Error processing paragraphs: {e}")

    return paragraphs_clean_list

# Example usage:
model_config = "phi3-38b-q4km"
base_truth_dir, base_sample_dir, base_score_dir, subdir_in = create_base_subdir(model_config)

# Replace this with your actual list of raw paragraphs
paragraphs_raw_list = [
    "The Way by Swann’s",
    "For Monsieur Gaston Calmette",
    "As a token of profound",
    "and affectionate gratitude.",
    "Marcel Proust",
    "PART I: Combray",
    "1",
    # Add more paragraphs as needed
]

# Process the paragraphs
paragraphs_clean_list = clean_paragraphs(paragraphs_raw_list, MIN_CHARS_PARAGRAPH, MIN_WORDS_PARAGRAPH)

# Print cleaned paragraphs for verification
for paragraph in paragraphs_clean_list:
    print(paragraph)
