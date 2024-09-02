import re
import os
import numpy as np
import pandas as pd
import joblib
from collections import Counter

def load_glove_model(glove_file):
    """Load GloVe model from file."""
    print(f"Loading GloVe model from {glove_file}...")
    model = {}
    with open(glove_file, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            vector = np.array(split_line[1:], dtype=np.float32)
            model[word] = vector
    print("Model loaded.")
    return model

def encode_text(text, glove_model):
    """Encode text into vectors using GloVe model."""
    words = text.lower().split()  # Simple tokenization (consider using a more advanced tokenizer)
    vectors = []
    for word in words:
        if word in glove_model:
            vectors.append(glove_model[word])
        else:
            print(f"Word '{word}' not found in GloVe model.")
    
    if vectors:
        return np.mean(vectors, axis=0)  # Return the average vector of all words
    else:
        # Return zero vector if no words found; use the vector size from the model
        return np.zeros(len(next(iter(glove_model.values()))), dtype=np.float32) 

def glove(train_df, test_df, vector_size, text_column='text'):
    """Process text data using GloVe embeddings."""
    
    # Load GloVe model
    glove_file = f'data/models/glove/{vector_size}/glove_{vector_size}.txt'
    if not os.path.exists(glove_file):
        raise FileNotFoundError(f"GloVe file not found: {glove_file}")
    
    glove_model = load_glove_model(glove_file)
    
    # Tokenize and encode text data
    train_df_transformed = train_df[text_column].apply(lambda x: encode_text(x, glove_model))
    test_df_transformed = test_df[text_column].apply(lambda x: encode_text(x, glove_model))
    
    # Convert encoded vectors to DataFrame
    train_vectors_df = pd.DataFrame(train_df_transformed.tolist())
    test_vectors_df = pd.DataFrame(test_df_transformed.tolist())

    # Concatenate original data with the TF-IDF features
    train_data = pd.concat(
        [train_df[["label"]].reset_index(drop=True), train_vectors_df], axis=1)
    test_data = pd.concat(
        [test_df[["label"]].reset_index(drop=True), test_vectors_df], axis=1)
    
    # Optionally, save the encoded vectors for both sets
    os.makedirs('data/vectors/glove', exist_ok=True)
    train_data.to_csv(f'data/vectors/glove/train_{vector_size}.csv', index=False)
    test_data.to_csv(f'data/vectors/glove/test_{vector_size}.csv', index=False)
    
    print(f"Encoded vectors saved for vector size {vector_size}")
