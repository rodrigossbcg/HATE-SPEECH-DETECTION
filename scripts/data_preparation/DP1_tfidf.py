import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def tfidf(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        vector_size: int,
        text_column: str = 'text'
) -> (pd.DataFrame, pd.DataFrame):

    model_path = f"data/models/tfidf/tfidf_{vector_size}.pkl"

    # Check if the model already exists
    if os.path.exists(model_path):
        # Load the existing model
        tfidf_vectorizer = joblib.load(model_path)
        print(f"Loaded existing TF-IDF model with {vector_size} dimensions.")
    else:
        # Initialize and fit a new model
        tfidf_vectorizer = TfidfVectorizer(max_features=vector_size)
        tfidf_vectorizer.fit(train_data[text_column])
        # Save the newly fitted model
        joblib.dump(tfidf_vectorizer, model_path)

    # Transform both training and testing data
    train_tfidf_matrix = tfidf_vectorizer.transform(train_data[text_column])
    test_tfidf_matrix = tfidf_vectorizer.transform(test_data[text_column])

    # Convert to DataFrame
    train_tfidf_df = pd.DataFrame(
        train_tfidf_matrix.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out()
    )

    test_tfidf_df = pd.DataFrame(
        test_tfidf_matrix.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out()
    )

    # Concatenate original data with the TF-IDF features
    train_data = pd.concat(
        [train_data[["label"]].reset_index(drop=True), train_tfidf_df], axis=1)
    test_data = pd.concat(
        [test_data[["label"]].reset_index(drop=True), test_tfidf_df], axis=1)

    # Save test and train data
    train_data.to_csv(
        f"data/vectors/tfidf/train_{vector_size}.csv", index=False)
    test_data.to_csv(
        f"data/vectors/tfidf/test_{vector_size}.csv", index=False)
