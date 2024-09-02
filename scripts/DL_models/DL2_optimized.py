from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, get_cosine_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from datasets import Dataset
import warnings
from transformers import logging as transformers_logging
import random
import torch
import json
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
            num_epochs=10
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
        # Metrics
        self.train_metrics = None
        self.eval_metrics = None

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

    def load_data(self, train_file_path, val_file_path):
        train_set = load_dataset('csv', data_files=train_file_path)['train']
        val_set = load_dataset('csv', data_files=val_file_path)['train']

        return self.prepare_data(train_set), self.prepare_data(val_set)

    def prepare_data(self, df):
        self.initialize_tokenizer()
        dataset = Dataset.from_pandas(df)
        return dataset.map(self.tokenize_and_align_labels, batched=True).remove_columns(['text']).with_format('torch')

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples['text'], truncation=True, padding=True)
        tokenized_inputs['labels'] = examples['label']
        return tokenized_inputs

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self, train_dataset, val_dataset, output_dir):

        total_steps = len(train_dataset) * self.num_epochs // self.batch_size

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps * self.warmup_ratio,
            num_training_steps=total_steps,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=os.path.join(output_dir, "logs"),
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            dataloader_num_workers=4,
            warmup_ratio=self.warmup_ratio,
            label_smoothing_factor=0.1,
            gradient_accumulation_steps=1,
            eval_strategy="steps",
            eval_steps=10,#total_steps // 25,
            logging_strategy="steps",
            logging_steps=10, #total_steps // 25,
            save_strategy="no",
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

    def save_metrics(self, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)
        train_file = os.path.join(output_dir, 'train_metrics.json')
        eval_file = os.path.join(output_dir, 'eval_metrics.json')
        
        with open(train_file, 'w') as f:
            json.dump(self.train_metrics, f, indent=2)
        
        with open(eval_file, 'w') as f:
            json.dump(self.eval_metrics, f, indent=2)
    
        print(f"Train metrics saved to: {train_file}")
        print(f"Eval metrics saved to: {eval_file}")

    def save_best_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")

def main():
    model_params = {
        "distilbert-base-uncased": {
            "model_name": "distilbert-base-uncased",
            "learning_rate": 2e-4,
            "weight_decay": 0.02,
            "dropout_rate": 0.3,
            "batch_size": 64
        },
        "albert-base-v2": {
            "model_name": "albert-base-v2",
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "dropout_rate": 0.1,
            "batch_size": 32
        },
        "roberta-base": {
            "model_name": "roberta-base",
            "learning_rate": 2e-5,
            "weight_decay": 0.05,
            "dropout_rate": 0.15,
            "batch_size": 24
        }
    }

    for model_name in model_params.keys():
        print(f"Training model: {model_name}")
        data_dir = "data/clean/"
        output_dir = f"results/DL/optimized/{model_name}/"
        os.makedirs(output_dir, exist_ok=True)

        classifier = BERTClassifier(**model_params[model_name])
        classifier.initialize_tokenizer()
        train_dataset, val_dataset = classifier.load_data(
            data_dir + "train_dataset.csv",
            data_dir + "test_dataset.csv"
        )
        classifier.train(train_dataset, val_dataset, output_dir=output_dir)
        classifier.save_metrics(f"{output_dir}/")
        classifier.save_best_model(f"{output_dir}/best_model")

if __name__ == "__main__":
    main()
