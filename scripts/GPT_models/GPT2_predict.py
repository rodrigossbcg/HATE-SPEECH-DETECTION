import openai
import json
import os
from dotenv import load_dotenv


class GPTEvaluator:

    def __init__(self, model_name, api_key, test_path, result_path, save=True):
        self.client = openai.Client(api_key=api_key)
        self.model_name = model_name
        self.test_path = test_path
        self.result_path = result_path
        self.save = save
        self.test_data = []
        self.results = []

    def _load_data(self):
        try:
            with open(self.test_path, 'r') as file:
                self.test_data = [json.loads(line) for line in file]
                print(f"Loaded {len(self.test_data)} test samples.")
        except Exception as e:
            raise e("Error: Unable to load test data from the file.")

    def _get_prompt(self, text):
        prompt_template = """
            You will be asigned a task:

            1) You will get one sentence as input.
            2) I want you to identify hatefull speech.
            3) Return ONLY the sentence with hatefull part CAPITALIZED.

            Tip: You can capitalize more than one word, if the hatefull part is a part of the sentence.

            Sentence: {text}
        """
        return prompt_template.format(text=text)

    def _completion(self, text):

        conversation = [
            {"role": "system", "content": "ChatGPT is an accurate hate speech detector."},
            {"role": "user", "content": self._get_prompt(text)}
        ]

        try:
            return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation).choices[0].message.content

        except Exception as e:
            raise e("Error: Unable to get completion from the model.")

    def _save_results(self):
        try:
            with open(self.result_path, "w") as file:
                json.dump(self.results, file, indent=4)
                print(f"Saved results to {self.result_path}")
        except Exception as e:
            raise e("Error: Unable to save results to the file.")

    def evaluate(self):

        if not self.test_data:
            self._load_data()

        for i, post in enumerate(self.test_data):

            if os.path.exists(self.result_path):
                print(f"File {self.result_path} already exists. Skipping save.")
                return

            # Extract the real text and output
            real_text = post['input']
            real_output = post['output']

            # Get the model's response
            try:
                pred_output = self._completion(real_text)

            except Exception as e: 
                print(f"Error: Document {i} - Unable to get completion from the model.")
                continue

            # Check if the arrays have different sizes
            if len(real_output.split()) != len(pred_output.split()):
                print(f"Error: Document {i} - Response does not match size.")

            else:
                self.results.append({
                    "real_output": real_output,
                    "pred_output": pred_output
                })
        
        if self.save:
            self._save_results()


if __name__ == "__main__":

    models = {"gpt-3.5": [
        "gpt-3.5-turbo-1106",
        "ft:gpt-3.5-turbo-1106:personal:hate-detector-0:9wvuSXIL",
        "ft:gpt-3.5-turbo-1106:personal:hate-detector-1:9ww2BAhp",
        "ft:gpt-3.5-turbo-1106:personal:hate-detector-2:9wwF4fkH",
        "ft:gpt-3.5-turbo-1106:personal:hate-detector-3:9wxKXCBZ",
        "ft:gpt-3.5-turbo-1106:personal:hate-detector-4:9wxRvNit",
        "ft:gpt-3.5-turbo-1106:personal:hate-detector-5:9xbgPF2m",
        "ft:gpt-3.5-turbo-1106:personal:hate-detector-6:9xbnJf7U",
        "ft:gpt-3.5-turbo-1106:personal:hate-detector-7:9xbuuptK",
        "ft:gpt-3.5-turbo-1106:personal:hate-detector-8:9xc12GEK",
        "ft:gpt-3.5-turbo-1106:personal:hate-detector-9:9xc8aq1n"
        ],
    "gpt-4o": [
        "gpt-4o-mini-2024-07-18",
        "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-0:9xXkvuPV",
        "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-1:9xXtxCm4",
        "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-2:9xY5VEac",
        "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-3:9xYHNcPc",
        "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-4:9xYPuTMM",
        "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-5:9xcux6sZ",
        "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-6:9xd6bDmc",
        "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-7:9xdIzPpr",
        "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-8:9xdbOeZ9",
        "ft:gpt-4o-mini-2024-07-18:personal:hate-detector-9:9xdnWatM"
        ]
    }

    for key in models:

        results_path = f"results/GPT/predictions/{key}"
        for model_name in models[key]:

            if model_name + ".json" not in os.listdir(results_path):
                print(f"Model: {model_name}")

                print("Evaluating the model:", model_name)
                evaluator = GPTEvaluator(
                    model_name=model_name,
                    api_key = os.getenv("OPENAI_API_KEY"),
                    test_path="data/GPT/test.jsonl",
                    result_path=f"{results_path}/{model_name}.json",
                    save=True
                )

                evaluator.evaluate()
