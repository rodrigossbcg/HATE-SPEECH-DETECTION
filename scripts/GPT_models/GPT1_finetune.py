import os
import time
import openai
import random
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class IncrementalFineTuner:

    def __init__(self, api_key, model_name, train_file_path, jump=0, batch_size=3, learning_rate_multiplier=1, n_epochs=10, suffix="hate-detector"):
        self.api_key = api_key
        self.model_name = model_name
        self.train_file_path = train_file_path
        self.batch_size = batch_size
        self.learning_rate_multiplier = learning_rate_multiplier
        self.n_epochs = n_epochs
        self.suffix = suffix
        self.jump = jump
        self.fine_tuned_model = None
        self.last_job_id = None

        # Authenticate the OpenAI client
        self.client = openai.Client(api_key=self.api_key)

        # Load the training data
        self.train_data = self._load_data()

    def _load_data(self):
        # Load the JSONL training data
        with open(self.train_file_path, "r") as file:
            lines = file.readlines()
        return lines

    def _get_random_samples(self, seed, n):
        random.seed(seed)
        return random.sample(self.train_data, n)

    def _upload_training_file(self, samples):
        
        temp_file_path = "data/GPT/temp_train.jsonl"

        # Write the samples to a temporary file
        with open(temp_file_path, "w") as temp_file:
            temp_file.writelines(samples)

        # Upload the training file
        file = self.client.files.create(
            file=open(temp_file_path, "rb"),
            purpose="fine-tune"
        )
        
        return file.id
    
    def _ft_is_available(self, job_id):
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return job.status == "succeeded"

    def fine_tune(self, iterations=5, jump=0):
        for iteration in range(jump, iterations):
            
            # Keep trying to create the fine-tuning job until it succeeds
            if self.last_job_id:

              not_finished = True

              while not_finished:
                  job = self.client.fine_tuning.jobs.retrieve(self.last_job_id)
                  not_finished = job.status != "succeeded"
                  print(f"  [{str(datetime.now())}] - Fine-tuning job now finished. Waiting for 1 minute before retrying.")
                  time.sleep(60)

              # Last model id
              self.fine_tuned_model = job.fine_tuned_model
              print("Last model id: ", self.fine_tuned_model)

            # Get random samples for this iteration
            samples = self._get_random_samples(iteration, 10)

            # Upload the training file
            file_id = self._upload_training_file(samples)

            # Attempt to create a new fine-tuning job
            job = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=self.fine_tuned_model if self.fine_tuned_model else self.model_name,
                hyperparameters={
                    "batch_size": self.batch_size,
                    "learning_rate_multiplier": self.learning_rate_multiplier,
                    "n_epochs": self.n_epochs
                },
                suffix=f"{self.suffix}-{iteration}",
                seed=iteration
            )

            # Define the last job ID
            self.last_job_id = job.id
            print("Last job id: ", self.last_job_id)

            # Wait for a few minutes before starting the next iteration
            print(f"  [{str(datetime.now())}] - Fine-tuning job now finished. Waiting for 5 minutes before retrying.")
            time.sleep(60 * 5)


def main(n, jump=0): 

  # params
  api_key = os.getenv("OPENAI_API_KEY")
  model_name = "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-4:9xYPuTMM"
  train_file_path = "data/GPT/train.jsonl"

  # Instantiate the class
  fine_tuner = IncrementalFineTuner(
    api_key,
    model_name,
    train_file_path,
    n_epochs=10,
    jump=jump,
    batch_size=3
  )

  # Run the fine-tuning process for a specified number of iterations
  fine_tuner.fine_tune(n, jump)


if __name__ == "__main__":
    main(10)
