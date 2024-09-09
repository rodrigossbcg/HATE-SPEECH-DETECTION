import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data/clean/dataset.csv')
data['text'] = data['text'].astype(str)
print("Number of sentences: ", len(data))

# Calculate sentence length percentiles by number of words
sentence_lengths = data['text'].apply(lambda x: len(x.split()))
percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
length_percentiles = sentence_lengths.quantile(percentiles)

print("Sentence length percentiles (in words):")
for p, length in zip(percentiles, length_percentiles):
    print(f"{p*100:.0f}th percentile: {length:.0f}")

# Minimum and maximum sentence length in words
print(f"Minimum sentence length: {sentence_lengths.min()} words")
print(f"Maximum sentence length: {sentence_lengths.max()} words")

# Average number of words per sentence
print(f"Average number of words per sentence: {sentence_lengths.mean():.2f}")

# Calculate probability of sentence length > 128 and > 64 words
prob_greater_128 = (sentence_lengths > 128).mean()
prob_greater_64 = (sentence_lengths > 64).mean()

print(f"Probability of sentence length > 128 words: {prob_greater_128:.4f}")
print(f"Probability of sentence length > 64 words: {prob_greater_64:.4f}")

# Absolute class distribution
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
