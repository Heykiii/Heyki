'''

from scrapegraphai.graphs import SmartScraperGraph

# Define the configuration for the graph
graph_config = {
    "llm": {
        "api_key": "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",
        "model": "gpt-3.5-turbo",
    },
}

# Create the SmartScraperGraph instance
smart_scraper_graph = SmartScraperGraph(
    prompt="List me all the text",
    source="https://www.coffeehouse.am/",
    config=graph_config
)

# Run the scraping pipeline
result = smart_scraper_graph.run()

# Print the extracted information
print(result)
'''


import requests
from bs4 import BeautifulSoup

def scrape_webpage(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the body of the HTML document
        body = soup.body

        # Extract text content from the body
        text = body.get_text(separator=' ')

        # Split text into individual words
        word_list = text.split()

        # Create tuples containing each word
        word_tuples = [(word,) for word in word_list]

        return word_tuples
    else:
        print("Failed to fetch the webpage. Status code:", response.status_code)
        return None

# URL of the webpage to scrape
url = 'https://myhealthycenter.com/'

# Scrape the webpage and get the text in a list of tuples
word_tuples = scrape_webpage(url)
lists = []
# Print the word tuples
for word_tuple in word_tuples:
    lists.append(word_tuple[0])


sentence = ' '.join(lists)






'''
էսի chat gpt նա 
import nltk
import string

# Download necessary NLTK data files
nltk.download('punkt')

def preprocess_data(text):
    sentences = nltk.sent_tokenize(text)
    data = []
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        new_sent = []
        for word in words:
            new_word = word.lower()
            if new_word[0] not in string.punctuation:
                new_sent.append(new_word)
        if len(new_sent) > 0:
            data.append(new_sent)
    return data

# Sample data
own_data = """
Dairy View entire section Butter, margarine Milk Non-dairy drink Condensed milk Kefir Tan, okroshka Matsoun Sour cream Eggs Curd products Yoghurts, creams Drinking yoghurt Yoghurt, pudding, fermented milk product Creams, cocktails Cheese Armenian cheese Imported cheese Cream Grocery View entire section Pasta Macaroni Vermicelli, noodles Spaghetti Lasagne, cannelloni Arishta, tatar-boraki Grains Buckwheat Bulgur, wheat grain Semolina, emmer wheat Quinoa, millet, rye bran Rice Flour Oil, ghee Oil Ghee Legumes Peas, chick-pea Beans, lentils Ketchup, mayonnaise, sauce, syrup Ketchup Sauce, syrup Mayonnaise Mustard, horseradish
"""

# Preprocess the data
processed_data = preprocess_data(own_data)
print(processed_data)

import gensim
import gensim.downloader as api

# Load pre-trained Word2Vec model
google_news_vectors = api.load('word2vec-google-news-300')

# Preprocess the data
processed_data = preprocess_data(own_data)

# Create a new Word2Vec model
model = gensim.models.Word2Vec(vector_size=300, window=5, min_count=1)
model.build_vocab(processed_data)

# Intersect the pre-trained vectors with the new model's vocabulary
# Align the new model with the pre-trained vectors
model.build_vocab([list(google_news_vectors.key_to_index.keys())], update=True)
model.wv.vectors_lockf = 1.0
model.wv.intersect_vectors(google_news_vectors, lockf=1.0)

# Train the model on your data
model.train(processed_data, total_examples=model.corpus_count, epochs=model.epochs)

# Save the model
model.save("custom_word2vec.model")

# Find words similar to 'tea'
print("Words similar to 'tea':")
similar_words = model.wv.most_similar("tea", topn=5)
for word in similar_words:
    print(word)
'''


