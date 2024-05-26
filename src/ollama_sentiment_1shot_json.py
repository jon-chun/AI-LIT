import requests
import json
from tqdm import tqdm

def get_ollama_sentiment(sentence, model_name="phi3"):
    """
    Call the Ollama local API to get the sentiment score for a sentence.
    :param sentence: The sentence to analyze.
    :param model_name: The name of the model to use for sentiment analysis.
    :return: Sentiment score as a float between -1.0 and 1.0.
    """
    url = "http://127.0.0.1:8000"  # Replace with the correct port if different
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": f"Analyze the sentiment of the following sentence and return a floating-point number between -1.0 (most negative), 0.0 (neutral) to 1.0 (most positive) in JSON format only like {{'sentiment':0.7}} with no other text in the response:\n\n\"{sentence}\"\n\n",
        "model": model_name
    }
    response = requests.post(f"{url}/v1/completions", headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        sentiment_response = response.json()
        return sentiment_response['choices'][0]['text']
    else:
        print(f"Failed to get sentiment from API. Status code: {response.status_code}, Response: {response.text}")
        return 0.0

# Example usage
if __name__ == "__main__":
    sentence_list = [
        "I love this product!",
        "This is the worst service ever.",
        "It's okay, not great but not terrible.",
        "I'm extremely happy with the results!",
        "I hate waiting in long lines."
    ]
    sentiment_scores = []
    for sentence in tqdm(sentence_list):
        sentiment_score = get_ollama_sentiment(sentence)
        sentiment_scores.append(sentiment_score)
    print(sentiment_scores)