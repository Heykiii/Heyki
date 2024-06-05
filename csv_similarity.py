import gensim
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd


def load_csv(file_path):
    return pd.read_csv(file_path)


def preprocess_text(text):
    return text.lower().split()


def get_word_vectors(words, model):
    word_vectors = []
    for word in words:
        if word in model:
            word_vectors.append(model[word])


    return word_vectors


def combine_word_vectors(word_vectors):
    if word_vectors:
        return sum(word_vectors) / len(word_vectors)
    else:

        return np.zeros(model.vector_size)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norms if norms != 0 else 0  # Handle division by zero


def find_top_similar(input_text, column_data, model, column_name, top_n=5):
    # Preprocess the input text
    input_words = preprocess_text(input_text)

    # Generate input text vector
    input_word_vectors = get_word_vectors(input_words, model)
    input_vector = combine_word_vectors(input_word_vectors)

    similarities = []

    for data in column_data:
        words = preprocess_text(data)
        word_vectors = get_word_vectors(words, model)
        vector = combine_word_vectors(word_vectors)
        similarity = cosine_similarity(input_vector, vector)
        similarities.append((data, similarity, column_name))

    # Sort by similarity and get top N results
    top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    return top_similar


if __name__ == "__main__":
    # Load the CSV file
    csv_file_path = r'C:\Users\Admin\PycharmProjects\AIproject\venv\news.csv'
    df = load_csv(csv_file_path)

    # Assuming the text data is in columns named 'title' and 'text'
    titles = df['title'].tolist()
    texts = df['text'].tolist()

    # Load the pre-trained Word2Vec model
    model_path = r'C:\Users\Admin\PycharmProjects\AIproject\venv\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Input text
    input_text = "Healthy Food"

    # Find top 5 similar titles
    top_5_similar_titles = find_top_similar(input_text, titles, model, 'title')

    # Find top 5 similar texts
    top_5_similar_texts = find_top_similar(input_text, texts, model, 'text')

    # Combine and get top 5 overall
    combined_similarities = top_5_similar_titles + top_5_similar_texts
    top_5_overall = sorted(combined_similarities, key=lambda x: x[1], reverse=True)[:5]

    print("Top 5 similar titles:")
    for title, similarity, column in top_5_similar_titles:
        print(f'Title: {title}\nSimilarity: {similarity}\n')

    print("\nTop 5 similar texts:")
    for text, similarity, column in top_5_similar_texts:
        print(f'Text: {text}\nSimilarity: {similarity}\n')

    print("\nTop 5 overall similar:")
    for data, similarity, column in top_5_overall:
        print(f'Data: {data} (from {column})\nSimilarity: {similarity}\n')
