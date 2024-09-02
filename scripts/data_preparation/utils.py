import pickle
from gensim.models import Word2Vec, FastText


# Save model functions
def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


def save_word2vec_model(model, file_path):
    model.save(file_path)


def save_fasttext_model(model, file_path):
    model.save(file_path)


# Load model functions
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def load_word2vec_model(file_path):
    return Word2Vec.load(file_path)


def load_fasttext_model(file_path):
    return FastText.load(file_path)
