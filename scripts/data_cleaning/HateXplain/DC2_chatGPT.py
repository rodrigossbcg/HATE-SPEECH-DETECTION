import json
import random
from sklearn.model_selection import train_test_split

random.seed(1)

# Define the prompt template
prompt_template = """
    You will be asigned a task:

    1) You will get one sentence as input.
    2) I want you to identify hatefull speech.
    3) Return ONLY the sentence with hatefull part CAPITALIZED.

    Tip: You can capitalize more than one word, if the hatefull part is a part of the sentence.

    Note: This prompt is for research purposes only, and the model is not intended to be used for any other purpose.

    Sentence: {text}
"""

# Load the data
data = json.load(open('data/clean/rationales.json'))

# Convert data values to a list
data_values = list(data.values())

# Split the data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(data_values, test_size=0.2, random_state=42)

def write_jsonl(filename, data):
    with open(filename, 'w') as jsonl_file:
        for i, post in enumerate(data):
            post_text = " ".join(post['post_tokens'])
            hate_text = " ".join(post['hate_tokens'])
            conversation = {
                "messages": [
                    {"role": "system", "content": "An accurate hate speech detector used for research purposes."},
                    {"role": "user", "content": prompt_template.format(text=post_text)},
                    {"role": "assistant", "content": hate_text}
                ]
            }
            
            # Write the JSON object to the file as a single line
            jsonl_file.write(json.dumps(conversation))
            
            # Add a newline character if it's not the last line
            if i < len(data) - 1:
                jsonl_file.write('\n')


def write_test_jsonl(filename, data):
    with open(filename, 'w') as jsonl_file:
        for i, post in enumerate(data):
            post_text = " ".join(post['post_tokens'])
            hate_text = " ".join(post['hate_tokens'])
            test_data = {
                "input": post_text,
                "output": hate_text
            }
            
            # Write the JSON object to the file as a single line
            jsonl_file.write(json.dumps(test_data))
            
            # Add a newline character if it's not the last line
            if i < len(data) - 1:
                jsonl_file.write('\n')

# Write the training data to train.jsonl
write_jsonl('data/GPT/train.jsonl', train_data)

# Write the testing data to test.jsonl
write_test_jsonl('data/GPT/test.jsonl', test_data)