import re

def split_text(text, chunk_size=299):
    paragraphs = re.split(r"\n\n+", text)
    sentences = []
    chunks = []
    
    for paragraph in paragraphs:
        paragraph_sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        sentences.extend(paragraph_sentences)
    
    for sentence in sentences:
        sentence_chunks = split_into_chunks(sentence, chunk_size)
        chunks.extend(sentence_chunks)
    
    return chunks