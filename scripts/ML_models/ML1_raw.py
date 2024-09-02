import pandas as pd
from sklearn.metrics import accuracy_score

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class ModelTrainer:

    def __init__(self, results_file="results/ML/csv/raw.csv"):
        self.results_file = results_file
        self.models = {
            'LREG': LogisticRegression,
            'LGBM': LGBMClassifier,
            'XGBoost': XGBClassifier,
            'NB': GaussianNB,
            'KNN': KNeighborsClassifier
        }
        self.results = self._load_results()

    def _load_results(self):
        try:
            results = pd.read_csv(self.results_file)
        except FileNotFoundError:
            results = pd.DataFrame(columns=["encoding_type", "vector_size", "model", "accuracy"])
        return results

    def train_and_evaluate(self, encoding_type, vector_size):
        # Load train and test data
        train = pd.read_csv(f"data/vectors/{encoding_type}/train_{vector_size}.csv")
        test = pd.read_csv(f"data/vectors/{encoding_type}/test_{vector_size}.csv")

        y_train = train["label"].astype(float)
        X_train = train.drop(columns=["label"])

        y_test = test["label"].astype(float)
        X_test = test.drop(columns=["label"])

        for key, model_class in self.models.items():
            if self._is_already_processed(encoding_type, vector_size, key):
                print(f"Skipping {encoding_type} - {vector_size} - {key}: already processed.")
                continue

            try:
                model = model_class()
                fit = model.fit(X_train, y_train)
                y_pred = fit.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                self._save_result(encoding_type, vector_size, key, accuracy)

                print(f"Success {encoding_type} - {vector_size} - {key}: [{round(accuracy, 2)}]")

            except Exception as e:
                print(f"Failure {encoding_type} - {vector_size} - {key}")

    def _is_already_processed(self, encoding_type, vector_size, key):
        return not self.results[
            (self.results["encoding_type"] == encoding_type) &
            (self.results["vector_size"] == vector_size) &
            (self.results["model"] == key)
        ].empty

    def _save_result(self, encoding_type, vector_size, key, accuracy):
        new_result = pd.DataFrame([[encoding_type, vector_size, key, accuracy]],
                                  columns=["encoding_type", "vector_size", "model", "accuracy"])
        self.results = pd.concat([self.results, new_result], ignore_index=True)
        self.results.to_csv(self.results_file, index=False)

    def run(self):
        for encoding_type in ["fasttext", "word2vec", "tfidf", "glove"]:
            vector_sizes = [300, 500, 1000] if encoding_type != "tfidf" else [3000, 5000, 10000]

            for vector_size in vector_sizes:
                self.train_and_evaluate(encoding_type, vector_size)


def main():
    trainer = ModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main()