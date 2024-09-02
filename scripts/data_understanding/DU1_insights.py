import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data/clean/dataset.csv')
data['text'] = data['text'].astype(str)
print("Number of sentences: ", len(data))

# Average number of words per sentence
data['num_words'] = data['text'].apply(lambda x: len(x.split()))
print(f"Average number of words per sentence: {data['num_words'].mean()}")

# Absolute lass distribution
print("Absolute distribution: ")
print(data['label'].value_counts())

# Relative class distribution
print("Relative class distribution: ")
print(data['label'].value_counts(normalize=True))

# Number of unique words
unique_words = set()
data['text'].str.split().apply(unique_words.update)
print(f"Number of unique words: {len(unique_words)}")


# Ensure 'text' column is string
data['text'] = data['text'].astype(str)
data = data.dropna(subset=['text'])

# Generate word clouds for each class
classes = data['label'].unique()
for c in classes:
    text = ' '.join(data[data['label'] == c]['text'])
    wordcloud = WordCloud(width=1000, height=1000, background_color='white').generate(text)
    print(f"Word cloud for class {c}")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
