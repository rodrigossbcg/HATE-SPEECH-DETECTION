import json
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, get_cosine_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from datasets import Dataset
import warnings
import random
import torch
import os

# Set environment variable to avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ignore specific warnings
warnings.filterwarnings("ignore", message="Some weights of DistilBertForSequenceClassification were not initialized")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model on a down-stream task")
warnings.filterwarnings("ignore", message="The current process just got forked")


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric = float('-inf')
        self.train_metrics = []
        self.eval_metrics = []

    def log(self, logs):
        if "eval_loss" in logs:
            self.eval_metrics.append(logs)
            eval_log = {k: v for k, v in logs.items() if k.startswith('eval_')}
            print(f"Eval metrics: {eval_log}", flush=True)
        elif "loss" in logs:
            self.train_metrics.append(logs)
            train_log = {k: v for k, v in logs.items() if not k.startswith('eval_')}
            print(f"Train metrics: {train_log}", flush=True)


class BERTClassifier:

    def __init__(
            self,
            model_name='distilbert-base-uncased',
            num_labels=2,
            batch_size=16,
            learning_rate=1e-4,
            weight_decay=0.1,
            dropout_rate=0.2,
            warmup_ratio=0.1,
            num_epochs=5,
            gradient_accumulation_steps=4,
            output_dir=""
        ):

        # Model parameters
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            ignore_mismatched_sizes=True
        )
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        # Training parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        # Metrics
        self.train_metrics = None
        self.eval_metrics = None
        self.data_dir = "data/clean/"
        self.output_dir = output_dir

        # Set random seed
        self.seed = 1
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        elif torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            clean_up_tokenization_spaces=True
        )

    def load_data(self, train_file_path, test_file_path):
        # Load the train dataset
        train_set = load_dataset('csv', data_files=train_file_path)['train']
        train_df = train_set.to_pandas()

        # Load the test dataset and split into validation and test sets (20% validation, 80% test)
        test_set = load_dataset('csv', data_files=test_file_path)['train']
        test_df = test_set.to_pandas()

        # Split test set into validation and test sets
        val_df, test_df = train_test_split(test_df, test_size=0.8, random_state=self.seed)

        train_dataset = self.prepare_data(train_df)
        val_dataset = self.prepare_data(val_df)
        test_dataset = self.prepare_data(test_df)

        print("Training data batch size:", len(train_dataset))
        print("Validation data batch size:", len(val_dataset))
        print("Test data batch size:", len(test_dataset))

        return train_dataset, val_dataset, test_dataset

    def prepare_data(self, df):

        if df['text'].isnull().any():
            df = df.dropna(subset=['text'])

        # Initialize tokenizer and prepare dataset
        self.initialize_tokenizer()
        dataset = Dataset.from_pandas(df)

        return dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            batch_size=len(dataset),
            remove_columns=['text']
        ).with_format("torch")

    def tokenize_and_align_labels(self, examples):
        # Ensure examples['text'] is a list of strings
        if not isinstance(examples['text'], list) or not all(isinstance(i, str) for i in examples['text']):
            raise ValueError("'text' field must be a list of strings")

        tokenized_inputs = self.tokenizer(
            examples['text'],  # This should be a list of strings
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors="pt"
        )

        return {k: v.squeeze(0) for k, v in tokenized_inputs.items()}

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        roc = roc_auc_score(labels, preds, average='weighted', multi_class='ovr')
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc
        }

    def train(self, train_dataset, val_dataset):

        total_steps = len(train_dataset) * self.num_epochs // (self.batch_size * self.gradient_accumulation_steps)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=os.path.join(self.output_dir, "logs"),
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            eval_strategy="steps",
            eval_steps=total_steps // 50,
            logging_strategy="steps",
            logging_steps=total_steps // 50,
            save_strategy="steps",
            save_steps=total_steps // 50,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_on_each_node=True,
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, scheduler)
        )

        trainer.train()
        self.train_metrics = trainer.train_metrics
        self.eval_metrics = trainer.eval_metrics

        # Save training history as JSON
        with open(f"{self.output_dir}/train_metrics.json", 'w') as f:
            json.dump(self.train_metrics, f, indent=2)

        # Save training history as JSON
        with open(f"{self.output_dir}/eval_metrics.json", 'w') as f:
            json.dump(self.eval_metrics, f, indent=2)

    def evaluate(self, test_dataset):
        # Load the best model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir + "/checkpoint")
        self.model.to(self.device)

        # Get predictions
        trainer = CustomTrainer(
            model=self.model,
            args=TrainingArguments(output_dir=self.output_dir),
            compute_metrics=self.compute_metrics
        )
        predictions = trainer.predict(test_dataset)
        metrics = self.compute_metrics(predictions)

        # Save results to CSV
        results_df = pd.DataFrame([metrics])
        results_df.to_csv(f"{self.output_dir}/test_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True, help="Mode: train or evaluate")
    args = parser.parse_args()

    model_names = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "GroNLP/hateBERT"]

    best_params = {
        "distilbert-base-uncased": {
            "batch_size": 64,
            "learning_rate": 0.0004, 
            "weight_decay": 0.08,
            "dropout_rate": 0.16
        },
        "roberta-base": {
            "batch_size": 16,
            "learning_rate": 0.00005, 
            "weight_decay": 0.025,
            "dropout_rate": 0.2
        },
        "bert-base-uncased": {
            "batch_size": 32,
            "learning_rate": 0.00016, 
            "weight_decay": 0.08,
            "dropout_rate": 0.15
        },
        "GroNLP/hateBERT": {
            "batch_size": 64,
            "learning_rate": 0.00022, 
            "weight_decay": 0.08,
            "dropout_rate": 0.26
        },
    }

    for model_name in model_names:
        folder_name = model_name.split('/')[-1].lower()
        print(f"Processing model: {folder_name}")
        output_dir = f"results/DL/optimized/{folder_name}/"
        os.makedirs(output_dir, exist_ok=True)

        classifier = BERTClassifier(
            model_name=model_name,
            output_dir=output_dir,
            **best_params[model_name])

        if args.mode == 'train':
            train_dataset, val_dataset, test_dataset = classifier.load_data(
                "data/clean/train_dataset_dl.csv",
                "data/clean/test_dataset_dl.csv"
            )
            # Train the model
            classifier.train(train_dataset, val_dataset)

        elif args.mode == 'evaluate':
            # Load the test dataset
            _, _, test_dataset = classifier.load_data(
                "data/clean/train_dataset_dl.csv",
                "data/clean/test_dataset_dl.csv"
            )
            # Evaluate the model
            classifier.evaluate(test_dataset)

if __name__ == "__main__":
    main()
