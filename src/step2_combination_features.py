# 
# Jon Chun
# 25 May 2024

# CLI Setup
# pip install vaderSentiment
# pip install -U textblob
# pip install -U textblob-fr
# pip install -q transformers



import os
import re
import shutil
import logging
import sys
import time
from tqdm import tqdm

# Sentiment: VADER (English Only)
# pip install vaderSentiment
# https://github.com/cjhutto/vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sentiment: TextBlob (Fr with extension)
# pip install -U textblob
# pip install -U textblob-fr
from textblob import TextBlob
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer

# Sentiment: Huggingface Transformers BERTMulti
# pip install -q transformers
from transformers import pipeline, AutoTokenizer

# Sentiment: Ollama LLMs Mistral 7B v0.3
# pip install ollama
# ollama run mistral:7b-instruct-v0.3-q4_K_M
import ollama
MODEL_OLLAMA = "mistral:7b-instruct-v0.3-q4_K_M"
MODEL_OLLAMA_SHORT = "mistral7b"

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
BOOK_VERSIONS_LIST = ['en_enright','en_davis','en_moncrieff','fr_proust']
BOOK_VERSION_SELECTION = BOOK_VERSIONS_LIST[0]

if BOOK_VERSION_SELECTION == "en_enright":

    # Swan's Way by Proust in En (trans. Enright)
    language_code = "en"
    subdir_title = "book_proust_en_swans-way_enright"
    filename_in =  "book_proust_en_swans-way_enright_sentence_clean.txt"
    filename_out = "book_proust_en_swans-way_enright_sentence_FEATURE.txt"

elif BOOK_VERSION_SELECTION == "en_moncrieff":

    # Swan's Way by Proust in En (trans. Moncrieff)
    language_code = "en"
    subdir_title = "book_proust_en_swans-way_moncrieff"
    filename_in  = "book_proust_en_swans-way_moncrieff_sentence_clean.txt"
    filename_out = "book_proust_en_swans-way_moncrieff_sentence_FEATURE.txt"

elif BOOK_VERSION_SELECTION == "en_davis":

    # Swan's Way by Proust in En (trans. Enright)
    language_code = "en"
    subdir_title = "book_proust_en_swans-way_davis"
    filename_in =  "book_proust_en_swans-way_davis_sentence_clean.txt"
    filename_out = "book_proust_en_swans-way_davis_sentence_FEATURE.txt"

elif BOOK_VERSION_SELECTION == "fr_proust":

    # Swan's Way by Proust in Fr by Proust
    language_code = "fr"
    subdir_title = "book_proust_fr_swans-way_proust"
    filename_in  = "book_proust_fr_swans-way_proust_sentence_clean.txt"
    filename_out = "book_proust_fr_swans-way_proust_sentence_FEATURE.txt"

else:

    print(f"ERROR: Invalid BOOK_VERSION_SELECTION: {BOOK_VERSION_SELECTION}")
    exit()





# Construct FULLPATHs to INPUT and OUTPUT files
FULLPATH_SEGMENTS_FILE_IN = os.path.join('..', 'data', 'step1_segments', 'book_proust_all_swans-way_human_validated', filename_in)
FULLPATH_SEGMENTS_FILE_OUT = os.path.join('..', 'data', 'step2_combination_features', filename_out)

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

def read_file_to_list(file_path):
    """
    Reads the content of the given file and returns it as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:

            # content = file.read()
            # content_clean = (" ").join(content.split())

            content_list = file.readlines()
            content_clean_list = [line.strip() for line in content_list]
            print(f"  DEBUG: START content_clean_list[:3]: {content_clean_list[:3]}")
            print(f"           END content_clean_list[-3:]: {content_clean_list[-3:]}")
            
            # content_clean_str = ' '.join(content_clean_list)
            # logging.info(f"Read {len(content_clean_str)} characters from the file: {file_path}")
    
            return content_clean_list
        
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



def get_sentiment_vader(sentence_list):
    """
    Given an input list of strings, returns an equal length list of the sentiment polarity values
    for corresponding strings using VADER.

    :param sentence_list: List of sentences (strings) to analyze.
    :return: List of sentiment polarity values.
    """
    # Initialize the VADER sentiment intensity analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Analyze the sentiment for each sentence in the list
    sentiment_scores = []
    for sentence in sentence_list:
        # Check if the sentence is not empty
        if sentence.strip():
            sentiment = analyzer.polarity_scores(sentence)
            # Append the compound score to the results list
            sentiment_scores.append(sentiment['compound'])
        else:
            # Append a neutral score for empty sentences
            sentiment_scores.append(0.0)

    return sentiment_scores


def get_sentiment_textblob(sentence_list, language="en"):

    sentiment_scores = []

    if language == "en":

        # Analyze the sentiment for each sentence in the list
        for sentence in tqdm(sentence_list, desc="Processing Sentences"):
            # Check if the sentence is not empty
            if sentence.strip():
                textblob_analysis = TextBlob(sentence)
                # Append the compound score to the results list
                sentiment_scores.append(textblob_analysis.sentiment.polarity)
            else:
                # Append a neutral score for empty sentences
                sentiment_scores.append(0.0)

    elif language == "fr":

        tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

        # Analyze the sentiment for each sentence in the list
        for sentence in sentence_list:
            # Check if the sentence is not empty
            if sentence.strip():
                sentiment_blob = tb(sentence)
                # Append the compound score to the results list
                sentiment_scores.append(sentiment_blob.sentiment)
            else:
                # Append a neutral score for empty sentences
                sentiment_scores.append(0.0)

    else:
        print(f"  ERROR: Invalid language for TextBlob: {language}")
        exit()

    return sentiment_scores


def convert_stars_to_int(star_string):
    """
    Converts a star rating string to an integer.

    :param star_string: A string representing the star rating (e.g., "3 stars").
    :return: An integer representing the star rating (1 to 5).
    """
    try:
        # Split the string to extract the numeric part
        parts = star_string.split()
        
        # Extract the first part and convert to integer
        star_int = int(parts[0])
        
        # Check if the integer is within the valid range
        if 1 <= star_int <= 5:
            return star_int
        else:
            raise ValueError("Star rating out of valid range (1 to 5).")
    except (ValueError, IndexError) as e:
        print(f"Error converting star string to int: {e}")
        return None  # or raise an exception, or handle it as per your requirements



def get_sentiment_bertmulti(sentence_list, language="en"):
    """
    Given an input list of strings, returns an equal length list of the sentiment polarity values
    for corresponding strings using VADER.

    :param sentence_list: List of sentences (strings) to analyze.
    :return: List of sentiment polarity values.
    """
    # Initialize the HF Transformer sentiment model
    hf_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    MAX_BERTMULTI_LEN = 350 # 512 is token limit * 3/4 word/token = 350 + ~50 padding
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    sentiment_bertmulti_pipeline = pipeline("sentiment-analysis", model=hf_model_name, tokenizer=tokenizer)

    # Analyze the sentiment for each sentence in the list
    sentiment_scores = []
    for sentence in tqdm(sentence_list, desc="BERT-Multi Sentiment Analysis"):
        # Trim sentence to MAX_BERTMULTI_LEN = 512
        sentence_trimmed = sentence[:MAX_BERTMULTI_LEN]
        if sentence_trimmed.strip():
            sentiment = sentiment_bertmulti_pipeline(sentence_trimmed)
            # Append the compound score to the results list
            # print(f" type(sentiment): {type(sentiment)}")
            # print(f" sentiment: {sentiment}")
            # print(f" dir(sentiment): {dir(sentiment)}")
            sentiment_star_int = convert_stars_to_int(sentiment[0]['label'])
            sentiment_scores.append(sentiment_star_int)
        else:
            # Append a neutral score for empty sentences
            sentiment_scores.append(0.0)

    return sentiment_scores

def get_sentiment_ollama(sentence_list, language="en", ollama_model=MODEL_OLLAMA):

    failure_count = 0
    sentiment_scores = []
    for sentence in tqdm(sentence_list, desc="Ollama Sentiment Analysis"):

        response = ollama.generate(
            model=ollama_model,
            # PROMPT #1: Directional Correctness for human validation of model results  
            # prompt=f"###SENTENCE:\n{sentence}\n\n###INSTRUCTIONS:\nGiven the above ###SENTENCE, estimate the sentiment as either 'negative', 'neutral', or 'positive' Return only one word for sentiment and nothing else, no header, explaination, introduction, summary, conclusion. Only return a single float number for the sentiment polarity"
            # PROMPT #2: Precise -1.0 to +1.0 sentiment for calcuation
            prompt=f"###SENTENCE:\n{sentence}\n\n###INSTRUCTIONS:\nGiven the above ###SENTENCE, estimate the sentiment as a float number from -1.0 (most negative) to 0.0 (neutral) to 1.0 (most positive). Return only one float number between -1.0 and 1.0   for sentiment polarity and nothing else, no header, explaination, introduction, summary, conclusion. Only return a single float number for the sentiment polarity"
        )

        sentiment_polarity = response['response'].strip()
        print(sentiment_polarity)
        print(f"type(sentiment_polarity): {type(sentiment_polarity)}")

        try:
            sentiment_polarity = float(sentiment_polarity)
            if sentiment_polarity > 1.0:
                sentiment_scores.append(1.0)
            elif sentiment_polarity < -1.0:
                sentiment_scores.append(-1.0)
            else:
                sentiment_scores.append(sentiment_polarity)
        except (ValueError, TypeError):
            # In case of error, default to 0.0
            failure_count += 1
            sentiment_scores.append(0.0)
    
    print(f"FAILURE COUNT: {failure_count}")
    print(f"FAILURE RATE: {(failure_count/len(sentence_list)):.2f}")    
    return sentiment_scores



def save_list_to_file(data_list, fullpath_fileout):
    """
    Saves a list of values, one per line, to the specified output file.

    :param data_list: List of values to be saved.
    :param fullpath_fileout: The path to the output file.
    """
    try:
        # Ensure the output directory exists
        output_directory = os.path.dirname(fullpath_fileout)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        # Write the values to the output file, one per line
        with open(fullpath_fileout, 'w', encoding='utf-8') as file:
            for item in data_list:
                file.write(f"{item}\n")
        
        print(f"Successfully saved data to {fullpath_fileout}")
    except IOError as e:
        print(f"Failed to write to file: {e}")
        raise

    return




def main():
    """
    Main function to process the segments file.
    """

    segments_list = []
    sentiment_vader_list = []

    # Use the command-line argument if provided, otherwise use the internal constant
    input_file_path = sys.argv[1] if len(sys.argv) > 1 else FULLPATH_SEGMENTS_FILE_IN

    # Check if the file exists
    if not os.path.exists(input_file_path):
        logging.error(f"File not found: {input_file_path}")
        print(f"File not found: {input_file_path}")
        return
    else:
        print(f"  input_file_path: {input_file_path}")

    # Determine the language based on the file path
    language_code = 'fr' if 'fr' in input_file_path.lower() else 'en'
    print(F"language_code: {language_code}")
    # Backup the file
    # backup_file(input_file_path)
    
    # Read input file into a text string
    segments_list = read_file_to_list(input_file_path)
    print(f" len(segments_list): {len(segments_list)}")

    """
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
    """;

    """
    # Get VADER Sentiment Sentiments
    sentiment_vader_list = get_sentiment_vader(segments_list)
    print(f"  sentiment_vader_list: {sentiment_vader_list}")

    # Get TextBlob Sentiment Sentiments
    sentiment_textblob_list = get_sentiment_textblob(segments_list)
    print(f"  sentiment_textblob_list: {sentiment_textblob_list}")

    # Get HF Transformers Model: nlptown/bert-base-multilingual-uncased-sentiment
    sentiment_bertmulti_list = get_sentiment_bertmulti(segments_list)
    print(f"  sentiment_bertmulti_list: {sentiment_bertmulti_list}")
    """;

    # Get Ollama LLM Sentiments
    sentiment_ollama_list = get_sentiment_ollama(segments_list)
    print(f"  sentiment_ollama_list: {sentiment_ollama_list}")


    # Save to output file
    """
    # Saving VADER sentiments
    fullpath_segments_feature_out = FULLPATH_SEGMENTS_FILE_OUT.replace("_FEATURE","_sentiment-vader")
    print(f"  fullpath_segments_feature_out SAVING to : {fullpath_segments_feature_out}")
    save_list_to_file(sentiment_vader_list, fullpath_segments_feature_out)

    # Saving TextBlob sentiments
    fullpath_segments_feature_out = FULLPATH_SEGMENTS_FILE_OUT.replace("_FEATURE","_sentiment-textblob")
    print(f"  fullpath_segments_feature_out SAVING to : {fullpath_segments_feature_out}")
    save_list_to_file(sentiment_textblob_list, fullpath_segments_feature_out)

    # Saving HF Transformers BertMulti sentiments
    fullpath_segments_feature_out = FULLPATH_SEGMENTS_FILE_OUT.replace("_FEATURE","_sentiment-bertmulti")
    print(f"  fullpath_segments_feature_out SAVING to : {fullpath_segments_feature_out}")
    save_list_to_file(sentiment_bertmulti_list, fullpath_segments_feature_out)
    """;
    # Saving Ollama LLM sentiments
    fullpath_segments_feature_out = FULLPATH_SEGMENTS_FILE_OUT.replace("_FEATURE",f"_sentiment-{MODEL_OLLAMA_SHORT}")
    print(f"  fullpath_segments_feature_out SAVING to : {fullpath_segments_feature_out}")
    save_list_to_file(sentiment_ollama_list, fullpath_segments_feature_out)


    # Process the segments file
    # sentiment_out_list = process_sentiment_file(file_text_str)
    # print(f"  RETURN from process_sentiment_file with len(sentiment_out_list): {len(sentiment_out_list)}")
    # 
    # write_sentences_to_file(FULLPATH_SEGMENTS_FILE_OUT, sentiment_out_list)

if __name__ == "__main__":
    main()
