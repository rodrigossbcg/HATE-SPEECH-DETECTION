# import scripts
from DP1_tfidf import *
from DP2_word2vec import *
from DP3_glove import *
from DP4_fasttext import *


# import packages
import os
import pandas as pd


# Inputs
train_path = "data/clean/train_dataset.csv"
test_path = "data/clean/test_dataset.csv"
vector_sizes = [300, 500, 1000, 3000, 5000, 10000]


# Create directories
os.makedirs('data/models', exist_ok=True)
os.makedirs('data/vectors', exist_ok=True)

embeddings = ['tfidf', 'word2vec', 'fasttext', 'glove']
for embedding in embeddings:
      os.makedirs(f"data/models/{embedding}", exist_ok=True)
      os.makedirs(f"data/vectors/{embedding}", exist_ok=True)


# Read data and select columns
print("Reading Database...")
train = pd.read_csv(train_path).dropna()
test = pd.read_csv(test_path).dropna()


# Generate embeddings for all vector sizes
for vector_size in vector_sizes:

    if vector_size <= 1000:
      
      # Word2Vec embedding
      print(f"Word2Vec Embedding S={vector_size}...")
      word2vec(train, test, vector_size)

      # FastText embedding
      print(f"FastText Embedding S={vector_size}...")
      fasttext(train, test, vector_size)

      # GloVe embedding
      print(f"GloVe Embedding S={vector_size}...")
      glove(train, test, vector_size)

    # TF-IDF embedding
    if vector_size > 1000:
      print(f"TF-IDF Embedding S={vector_size}...")
      tfidf(train, test, vector_size)

print("Data Preparation Done!")
