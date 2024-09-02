import os
import numpy as np
import pandas as pd
from gensim.models import FastText


def fasttext(train_df, test_df, vector_size, text_column='text'):
    """
    Train a FastText model on the training set if not already trained, encode both training and testing sets,
    and save the model and encoded vectors.

    Parameters:
    train_df (pd.DataFrame): DataFrame containing the training text data.
    test_df (pd.DataFrame): DataFrame containing the testing text data.
    vector_size (int): The size of the word vectors.
    text_column (str): The column name containing the text data.
    model_path (str): Path to save or load the FastText model.

    Returns:
    pd.DataFrame, pd.DataFrame: Modified DataFrames with the text column replaced by vector columns for both training and testing sets.
    """

    model_path = f"data/models/fasttext/fasttext_{vector_size}.model"

    # Function to tokenize text data
    def tokenize_text(df):
        return df[text_column].apply(lambda x: x.split())

    # Function to generate word vectors for a set of sentences
    def generate_word_vectors(sentences, model):
        word_vectors = []
        for sentence in sentences:
            vectors = [model.wv[word] for word in sentence if word in model.wv]
            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(vector_size)
            word_vectors.append(doc_vector)
        return pd.DataFrame(word_vectors, columns=[f'vector_{i}' for i in range(vector_size)])

    # Check if the model already exists
    if os.path.exists(model_path):
        # Load the existing model
        fasttext_model = FastText.load(model_path)
        print(f"Loaded existing FastText model with {vector_size} dimensions.")
    else:
        # Train the FastText model on the training set
        train_sentences = tokenize_text(train_df)
        fasttext_model = FastText(
            train_sentences, vector_size=vector_size, window=5, min_count=1, workers=4)
        # Save the model
        fasttext_model.save(model_path)

    # Encode both training and testing sets
    train_sentences = tokenize_text(train_df)
    test_sentences = tokenize_text(test_df)
    train_vectors = generate_word_vectors(train_sentences, fasttext_model)
    test_vectors = generate_word_vectors(test_sentences, fasttext_model)

    # Concatenate the new vector columns with the original DataFrames (excluding the text column)
    train_df_transformed = pd.concat(
        [train_df.drop(columns=[text_column]), train_vectors], axis=1)
    test_df_transformed = pd.concat(
        [test_df.drop(columns=[text_column]), test_vectors], axis=1)

    # Optionally, save the encoded vectors for both sets
    train_df_transformed.to_csv(
        f'data/vectors/fasttext/train_{vector_size}.csv', index=False)
    test_df_transformed.to_csv(
        f'data/vectors/fasttext/test_{vector_size}.csv', index=False)
