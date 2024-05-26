import os
import re
import shutil
import logging
import sys
import time

import nltk
# nltk.download('punkt')

import spacy
from spacy.pipeline import Sentencizer
# Load a blank SpaCy model
# nlp = spacy.load("en_core_web_md")
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
# Increase the maximum length limit
nlp.max_length = 2000000  # Adjust as needed

# Add the sentencizer to the pipeline
sentencizer = nlp.add_pipe("sentencizer")

import pysbd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Internal constant for the default path to the input file

# Swan's Way by Proust in En (trans. Moncrieff)
subdir_title = "book_proust_en_swans-way_moncrieff"
filename_in  = "book_proust_en_swans-way_moncrieff_sentence_clean.txt"
filename_out = "book_proust_en_swans-way_moncrieff_sentence_sentiment.txt"

# Swan's Way by Proust in Fr by Proust
subdir_title = "book_proust_fr_swans-way_proust"
filename_in  = "book_proust_fr_swans-way_proust_sentence_clean.txt"
filename_out = "book_proust_fr_swans-way_proust_sentence_sentiment.txt"
# Swan's Way by Proust in En (trans. Enright)
subdir_title = "book_proust_en_swans-way_enright"
filename_in =  "book_proust_en_swans-way_enright_sentence_clean.txt"
filename_out = "book_proust_en_swans-way_enright_sentence_sentiment.txt"
# Swan's Way by Proust in En (trans. Enright)
subdir_title = "book_proust_en_swans-way_davis"
filename_in =  "book_proust_en_swans-way_davis_sentence_clean.txt"
filename_out = "book_proust_en_swans-way_davis_sentence_sentiment.txt"


# Construct FULLPATHs to INPUT and OUTPUT files
FULLPATH_SEGMENTS_FILE_IN = os.path.join('..', 'data', 'step1_segments', 'book_proust_all_swans-way_human_validated', filename_in)
FULLPATH_SEGMENTS_FILE_OUT = os.path.join('..', 'data', 'step2_sentiments', 'book_proust_all_swans-way_human_validated', filename_out)

def backup_file(file_path):
    """
    Creates a backup of the given file in the same directory with a modified suffix.
    """
    try:
        # Create a backup file path
        directory, filename = os.path.split(file_path)
        base, ext = os.path.splitext(filename)
        backup_filename = f"{base}_backup__{ext}"
        backup_path = os.path.join(directory, backup_filename)

        # Copy the file to the backup path
        shutil.copy2(file_path, backup_path)
        logging.info(f"Backup created at: {backup_path}")
    except Exception as e:
        logging.error(f"Failed to create backup: {e}")
        raise

def read_file(file_path):
    """
    Reads the content of the given file and returns it as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:

            # content = file.read()
            # content_clean = (" ").join(content.split())

            content_list = file.readlines()
            content_clean_list = [line.strip() for line in content_list]
            print(f"  DEBUG: content_clean_list[:500]: {content_clean_list[:500]}")
            
            content_clean_str = ' '.join(content_clean_list)
            logging.info(f"Read {len(content_clean_str)} characters from the file: {file_path}")
    
            return content_clean_str
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise
    except IOError as e:
        logging.error(f"Failed to read file: {e}")
        raise

def write_sentences_to_file(file_path, sentences_list):
    """
    Writes non-blank sentences to the given file, one per line.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            non_blank_sentences = [sentence.strip() for sentence in sentences_list if sentence.strip()]
            file.write('\n'.join(non_blank_sentences))
            logging.info(f"Successfully wrote {len(non_blank_sentences)} sentences to the file: {file_path}")
    except IOError as e:
        logging.error(f"Failed to write to file: {e}")
        raise


# METHOD #1: NLTK w/punkt
def segment_text_nltk(text):
    from nltk.tokenize import sent_tokenize
    sentences_list = sent_tokenize(text)
    return sentences_list

# METHOD #2: SpaCy
def segment_text_spacy(text):
    doc = nlp(text)
    sentences_list = [sent.text for sent in doc.sents]
    return sentences_list

    # nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "lemmatizer"])
    # doc = nlp(text)
    # sentences_list = [sent.text for sent in doc.sents]
    # return sentences_list

def split_into_sentences(text):
    doc = nlp(text)
    sentences_list = [sent.text for sent in doc.sents]
    return sentences_list

# METHOD #3: RegEx
def segment_text_regex(text, language="en"):
    # TODO: language not used
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences_list = sentence_endings.split(text)
    return sentences_list

# METHOD #4: PySBD
def segment_text_pysbd(text, language='en'):
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences_list = segmenter.segment(text)
    return sentences_list




def process_sentiment_file(textfile_str, language_code='en'):
    """
    Processes the given file by segmenting its text into sentences and writing them back to the file.
    """
    sentences_ls = []
    sentiment_vader_list = []
    sentences_spacy_list = ""
    sentences_regex_list = ""
    sentences_pysbd_list = ""
    try:
        # METHOD #1: (NLTK/punkt)
        start_time = time.time()
        sentences_nltk_list = segment_text_nltk(textfile_str)
        print(f"The execution took {time.time() - start_time} seconds.")

        print(f"  #1 NLTK gives {len(sentences_nltk_list)} sentences")

        # METHOD #2: (SpaCy)
        start_time = time.time()
        # sentences_spacy_list = segment_text_spacy(textfile_str)
        # sentences_spacy_list = segment_text_spacy(textfile_str)
        print(f"The execution took {time.time() - start_time} seconds.")

        print(f"  #2 SpaCy gives {len(sentences_spacy_list)} sentences")

        # METHOD #3: (RegEx) simple fast RegEx split of very long char string (1-1.5M for Proust's Swan's Way)
        start_time = time.time()
        sentences_regex_list = segment_text_regex(textfile_str, language_code)
        print(f"The execution took {time.time() - start_time} seconds.")

        print(f"  #3 RegEx gives {len(sentences_regex_list)} sentences")

        # METHOD #4: (PySBD) = for 1.1M char, this took over 10mins and never finished on 32GB HP Victus NVIDIA 3060 gaming laptop
         # Initialize the PySBD segmenter
        seg = pysbd.Segmenter(language=language_code, clean=False)       
        start_time = time.time()
        # sentences_pysbd_list = segment_text_pysbd(textfile_str)
        print(f"The execution took {time.time() - start_time} seconds.")

        print(f" SEGMENT by PySBD gives {len(sentences_pysbd_list)} sentences")

        # Pick Method with the fewest sentences

        # Combine the lists into one list of lists
        all_lists = [sentences_nltk_list, sentences_spacy_list, sentences_regex_list, sentences_pysbd_list]

        # Find the list with the fewest strings
        sentences_fewest_list = min(all_lists, key=len)

        # Get the Name (not value) fo the list with the fewest elements
        # Find the name of the variable that is the shortest list
        shortest_list_var_name = None
        for name, value in locals().items():
            if value is sentences_fewest_list:
                shortest_list_var_name = name
                break

        print(f"  {shortest_list_var_name} is the method producing the most agglomerated sentences with a total of {len(sentences_fewest_list)}")

        # Concatenate its elements into a single string
        sentences_shortest_str = ' '.join(sentences_fewest_list)

        logging.info(f"Segmented the text into {len(sentences_regex_list)} sentences.")

        # return sentences_shortest_str
        return sentences_regex_list



    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        raise


def main():
    """
    Main function to process the segments file.
    """

    sentences_out_list = []
    # Use the command-line argument if provided, otherwise use the internal constant
    input_file_path = sys.argv[1] if len(sys.argv) > 1 else FULLPATH_SEGMENTS_FILE_IN

    # Check if the file exists
    if not os.path.exists(input_file_path):
        logging.error(f"File not found: {input_file_path}")
        print(f"File not found: {input_file_path}")
        return

    # Determine the language based on the file path
    language_code = 'fr' if 'fr' in input_file_path.lower() else 'en'

    # Backup the file
    # backup_file(input_file_path)
    
    # Read input file into a text string
    file_text_str = read_file(input_file_path)

    # Replace alphanum+punct word combinations that cause segmentation problems
    word_replacement_dict = {
        " M. ": " M ",
        " Mme. ": " Mme "
    }
    # Assuming `problematic_words` is a list of the keys in `word_replacement_dict`
    word_replacement_key_list = list(word_replacement_dict.keys())

    for word in word_replacement_key_list:
        if word in word_replacement_dict:
            file_text_str = file_text_str.replace(word, word_replacement_dict[word])

    # for word_problem_now in word_replacement_key_list:
    #     file_text_str = file_text_str.replace(word_problem_now, word_problem_dict[word_problem_now])

    # Process the segments file
    sentiment_out_list = process_sentiment_file(file_text_str)
    print(f"  RETURN from process_sentiment_file with len(sentiment_out_list): {len(sentiment_out_list)}")
    # 
    write_sentences_to_file(FULLPATH_SEGMENTS_FILE_OUT, sentiment_out_list)

if __name__ == "__main__":
    main()
