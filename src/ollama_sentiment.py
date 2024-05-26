import ollama


sentence_list = [
    "I love this product!",
    "This is the worst service ever.",
    "It's okay, not great but not terrible.",
    "I'm extremely happy with the results!",
    "I hate waiting in long lines."
]

model_small = "phi3:latest"
model_bilingual = "mistral:7b-instruct-v0.3-q4_K_M"


for sentence_index, sentence_now in enumerate(sentence_list):



    response = ollama.generate(
        model=model_bilingual,
        prompt=f"###SENTENCE:\n{sentence_now}\n\n###INSTRUCTIONS:\nGiven the above ###SENTENCE, estimate the sentiment as either 'negative', 'neutral', or 'positive' Return only one word for sentiment and nothing else, no header, explaination, introduction, summary, conclusion. Only return a single float number for the sentiment polarity"
    )
    print(response['response'])
    
    """
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': f"###SENTENCE:\n{sentence_now}\n\n###INSTRUCTIONS:\nGiven the above ###SENTENCE, estimate the sentiment as either 'negative', 'neutral', or 'positive' Return only one word for sentiment and nothing else, no header, explaination, introduction, summary, conclusion. Only return a single float number for the sentiment polarity",
        }
    ])
    print(response['message']['content'])
    """;

