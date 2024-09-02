import json

def process_data(data):
    initial_count = len(data)
    print(f"Initial number of observations: {initial_count}")

    filtered_data = {}
    hatespeech_count = 0
    rationale_match_count = 0

    for post_id, post_info in data.items():
        annotators = post_info['annotators']
        rationales = post_info['rationales']
        post_tokens = post_info['post_tokens']

        # Only consider posts where all annotators agree on the hatespeech label
        if all(annotator['label'] == 'hatespeech' for annotator in annotators):
            hatespeech_count += 1

            # Only consider annotations where all annotators agree on the rationale
            if all(rationales[0] == rationale for rationale in rationales):
                if len(rationales[0]):
                    rationale_match_count += 1

                # Create a new list of tokens where the tokens in the rationale are uppercased
                hate_tokens = [
                    post_tokens[i].upper() if rationales[0][i] == 1
                    else post_tokens[i].lower()
                    for i in range(len(post_tokens))
                ]

                # Add the filtered document with the new "hate_tokens" field
                filtered_data[post_id] = {
                    'post_tokens': post_tokens,
                    'hate_tokens': hate_tokens
                }

    print(f"Number of observations after removing non-hatespeech labels: {hatespeech_count}")
    print(f"Number of observations after removing non-matching rationales: {rationale_match_count}")

    return filtered_data

# Load the data
with open('data/raw/rationales.json') as f:
    data = json.load(f)
    print("Data loaded successfully.")
    print(f"Total number of posts: {len(data)}")

# Process the data
filtered_data = process_data(data)

# Save the filtered data to a new JSON file
with open('data/clean/rationales.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)
    print("Data cleaning completed successfully.")
    print(f"Total number of filtered posts: {len(filtered_data)}")
