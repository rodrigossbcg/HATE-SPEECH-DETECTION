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
import joblib
import os
import optuna
from optuna.trial import TrialState

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
            num_epochs=5
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
        self.data_dir = "data/clean/"
        self.output_dir = "results/DL/finetune/"

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

        train_df = train_set.to_pandas()
        train_df_sampled = train_df.sample(1800, random_state=self.seed).reset_index(drop=True)

        val_df = val_set.to_pandas()
        val_df_sampled = val_df.sample(n=200, random_state=self.seed).reset_index(drop=True)

        train_dataset = self.prepare_data(train_df_sampled)
        val_dataset = self.prepare_data(val_df_sampled)

        print("Training data batch size:", len(train_dataset))
        print("Validation data batch size:", len(val_dataset))

        return train_dataset, val_dataset

    def prepare_data(self, df):
        self.initialize_tokenizer()
        dataset = Dataset.from_pandas(df)
        # Use batched=True for faster processing
        return dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            batch_size=len(dataset),
            remove_columns=['text']
        ).with_format("torch")

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=512,  # You can adjust this value based on your needs
            return_tensors="pt"
        )
        # Convert to regular tensors (not batched)
        tokenized_inputs = {k: v.squeeze(0) for k, v in tokenized_inputs.items()}
        tokenized_inputs['labels'] = torch.tensor(examples['label'])
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

    def train(self, train_dataset, val_dataset):

        # Before the training loop
        print("Training data batch size:", len(train_dataset))
        print("Validation data batch size:", len(val_dataset))

        # Checking the shape of a sample batch
        sample_batch = train_dataset[0]
        print("Sample input shape:", sample_batch['input_ids'].shape)
        print("Sample attention mask shape:", sample_batch['attention_mask'].shape)
        print("Sample label shape:", sample_batch['labels'].shape)

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
            output_dir=self.output_dir,
            logging_dir=os.path.join(self.output_dir, "logs"),
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=1,
            eval_strategy="steps",
            eval_steps=total_steps // 10,
            logging_strategy="steps",
            logging_steps=total_steps // 50,
            save_strategy="no"
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

    def objective(self, trial):

        # Suggest values for each hyperparameter
        self.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        self.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
        self.weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-1)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)

        # Update model configuration with new dropout rate
        self.config.hidden_dropout_prob = dropout_rate
        self.config.attention_probs_dropout_prob = dropout_rate
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=self.config,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)

        train_dataset, val_dataset = self.load_data(
            f"{self.data_dir}train_dataset.csv",
            f"{self.data_dir}test_dataset.csv"
        )
        self.train(train_dataset, val_dataset)

        best_f1 = max(metric['eval_f1'] for metric in self.eval_metrics)
        return best_f1

def main(n_trials=20):

    for model_name in ["distilbert-base-uncased", "albert-base-v2", "roberta-base"]:
        print(f"Optimizing hyperparameters for model: {model_name}")
        output_dir = f"results/DL/finetune/{model_name}/"
        os.makedirs(output_dir, exist_ok=True)
        classifier = BERTClassifier(model_name=model_name)
        study = optuna.create_study(direction="maximize")
        study.optimize(classifier.objective, n_trials=n_trials)
        joblib.dump(study, output_dir + "study.pkl")

if __name__ == "__main__":
    main(n_trials=25)
