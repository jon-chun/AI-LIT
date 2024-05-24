import os
import shutil
import logging
import sys
import pysbd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Internal constant for the default path to the input file
filename_in = "book_proust_en_swans-way_moncrieff_sentence_raw.txt"
subdir_title = "book_proust_en_swans-way_moncrieff"


# FULLPATH_SEGMENTS_FILE = os.path.join('..', 'data', 'step1_segments', 'book_title_trans', 'book_title_translator.txt')
FULLPATH_SEGMENTS_FILE = os.path.join('..', 'data', 'books', 'step1_segments', subdir_title, filename_in)

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
            content = file.read()
            logging.info(f"Read {len(content)} characters from the file: {file_path}")
            return content
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise
    except IOError as e:
        logging.error(f"Failed to read file: {e}")
        raise

def write_sentences_to_file(file_path, sentences):
    """
    Writes non-blank sentences to the given file, one per line.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            non_blank_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            file.write('\n'.join(non_blank_sentences))
            logging.info(f"Successfully wrote {len(non_blank_sentences)} sentences to the file: {file_path}")
    except IOError as e:
        logging.error(f"Failed to write to file: {e}")
        raise

def process_segments_file(file_path, language='en'):
    """
    Processes the given file by segmenting its text into sentences and writing them back to the file.
    """
    try:
        # Step 1: Backup the file
        backup_file(file_path)

        # Step 2: Read the file content
        text = read_file(file_path)

        # Step 3: Initialize the PySBD segmenter
        seg = pysbd.Segmenter(language=language, clean=False)

        # Step 4: Segment the text into sentences
        sentences = seg.segment(text)
        logging.info(f"Segmented the text into {len(sentences)} sentences.")

        # Step 5: Write the sentences back to the original file
        write_sentences_to_file(file_path, sentences)

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        raise

def main():
    """
    Main function to process the segments file.
    """
    # Use the command-line argument if provided, otherwise use the internal constant
    input_file_path = sys.argv[1] if len(sys.argv) > 1 else FULLPATH_SEGMENTS_FILE

    # Check if the file exists
    if not os.path.exists(input_file_path):
        logging.error(f"File not found: {input_file_path}")
        print(f"File not found: {input_file_path}")
        return

    # Determine the language based on the file path
    language = 'fr' if 'fr' in input_file_path.lower() else 'en'

    # Process the segments file
    process_segments_file(input_file_path, language)

if __name__ == "__main__":
    main()
