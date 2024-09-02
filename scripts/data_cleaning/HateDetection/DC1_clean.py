from nltk.corpus import stopwords
import contractions
import pandas as pd
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def recursive_clean(text):
    # Remove leading and trailing spaces, quotes, signs, and usernames
    text_clean = text.strip()
    text_clean = re.sub(r'^@user:?', '', text_clean)
    if text_clean != text:
        return recursive_clean(text_clean)
    return text


def remove_emojis(text):

    text = re.sub(r'&[#A-Za-z0-9]+;', '', text)

    # Remove Unicode emojis and symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    return text


#############
# LOAD DATA #
#############

# Read dataset
data = pd.read_csv(f'data/raw/dataset.csv')

# Rename and select columns
data.rename(columns={'Content': 'text', 'Label': 'label'}, inplace=True)
data = data[['text', 'label']]

# Filter out mistakes in the label column
data = data[data['label'] != "Label"]

# Convert the label column to integers
data['label'] = data['label'].astype(int)

###############
# REMOVE TEXT #
###############

# Remove rows with missing text
data = data.dropna()
print("After drop NA's:",  len(data))

# Remove duplicates based on the 'text' column
data = data.drop_duplicates(subset=['text'])
print("After drop duplicates:",  len(data))

# Delete posts that have an URL
data = data[~data['text'].str.contains('http[s]?://')]
print("After removing URLs:",  len(data))


##############
# CLEAN TEXT #
##############

# Anonymize usernames
data['text'] = data['text'].apply(lambda x: re.sub(r'@\w+', '@user', x))

# Unify user format
data['text'] = data['text'].apply(lambda x: x.replace("<user>", "@user"))

# Convert to lower case
data['text'] = data['text'].str.lower()

# Replace "rt" with ""
data['text'] = data['text'].str.replace('rt', '')

# Remove multiple spaces
data['text'] = data['text'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Remove emojis
data['text'] = data['text'].apply(lambda x: remove_emojis(x))

# Expand contractions
data['text'] = data['text'].apply(lambda x: contractions.fix(x))

# Remove unwanted characters (punctuation)
data['text'] = data['text'].apply(lambda x: re.sub(r'[^\w\s@]', ' ', x))

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(
    lambda x:
    ' '.join([
        word for word in x.split()
        if word.lower() not in stop_words
    ])
)

# Remove leading and trailing spaces, quotes, signs, and usernames (recursive)
data['text'] = data['text'].apply(lambda x: recursive_clean(x))


###########################
# REMOVE TEXT AFTER CLEAN #
###########################

# Remove rows with missing text
data = data.dropna()
print("Drop missing values (after clean):",  len(data))

# Remove duplicates based on the 'text' column
data = data.drop_duplicates(subset=['text'])
print("Drop duplicates (after clean):",  len(data))


##########################
# TOKENIZE AND LEMMATIZE #
##########################

# Tokenize the text into words
data["text"] = data["text"].apply(word_tokenize)

# Normalization (lemmatization)
lemmatizer = WordNetLemmatizer()
data["text"] = data["text"].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x])
)


###########################
# REMOVE TEXT AFTER CLEAN #
###########################

# Remove rows with missing text
data = data.dropna()
print("Drop missing values (after tokenize):",  len(data))

# Remove duplicates based on the 'text' column
data = data.drop_duplicates(subset=['text'])
print("Drop duplicates (after tokenize):",  len(data))


#############
# SAVE DATA #
#############

# Save the modified dataframe to a new CSV file
data.to_csv('data/clean/dataset.csv', index=False)

# Print the length of the data after processing
print(f"\nData length after processing: {len(data)}")

# Count the value counts of the 'label' column
print(round(data["label"].value_counts() / len(data) * 100, 2), "\n")
