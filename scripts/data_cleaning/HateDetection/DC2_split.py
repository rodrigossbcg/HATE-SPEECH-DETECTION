import pandas as pd
from sklearn.model_selection import train_test_split

suffix = "_dl"
# Load and preprocess the data
data = pd.read_csv(f"data/clean/dataset{suffix}.csv")
print(set(data["label"]))
data["label"] = data["label"].astype(int)
print(len(data))

# Separate majority and minority labeles
majority_label = data[data['label'] == 0]
print(len(majority_label))
minority_label = data[data['label'] == 1]
print(len(minority_label))

# Downsample majority label
majority_label_undersampled = majority_label.sample(len(minority_label), random_state=42)
print(len(majority_label_undersampled))

# Combine minority label with undersampled majority label
undersampled_data = pd.concat([majority_label_undersampled, minority_label])
print(undersampled_data["label"].value_counts())

# Split the data into training and testing sets
X = undersampled_data.drop(columns=["label"])
y = undersampled_data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and target for saving
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the training and testing sets to CSV files
train_data.to_csv(f"data/clean/train_dataset{suffix}.csv", index=False)
test_data.to_csv(f"data/clean/test_dataset{suffix}.csv", index=False)

# Save the corpus to a text file
corpus_path = f'data/clean/corpus{suffix}.txt'
with open(corpus_path, 'w') as file:
    for line in X_train['text']:
        file.write(f"{line}\n")

# Print the size of the resulting datasets
print(f"Train set: {len(train_data)} samples")
print(f"Test set: {len(test_data)} samples")
