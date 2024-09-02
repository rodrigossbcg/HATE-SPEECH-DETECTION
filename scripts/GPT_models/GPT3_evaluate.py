import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re


def compare_words(real_word, pred_word):
    # Check if the words have the same length
    if len(real_word) != len(pred_word):
        return False
    
    # Iterate through each character index in both words
    for real_char, pred_char in zip(real_word, pred_word):
        # Check if characters are different and neither is a wildcard (non-alphanumeric)
        if real_char != pred_char and not re.match(r'\W', pred_char) and not re.match(r'\W', real_char):
            return False
    return True


def calculate_average_accuracy(real_output, pred_output):
    real_words = real_output.split()
    pred_words = pred_output.split()
    total_words = len(real_words)
    matches = 0

    for real_word, pred_word in zip(real_words, pred_words):
        if compare_words(real_word, pred_word):
            matches += 1

    return (matches / total_words) * 100 if total_words > 0 else 0


def calculate_average_precision(real_output, pred_output):
    real_words = real_output.split()
    pred_words = pred_output.split()
    
    tp = 0  # True Positives
    fp = 0  # False Positives

    for real_word, pred_word in zip(real_words, pred_words):
        if real_word.isupper() and pred_word.isupper():
            tp += 1
        elif real_word.islower() and pred_word.isupper():
            fp += 1

    # Precision = TP / (TP + FP)
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    return precision


def calculate_average_recall(real_output, pred_output):
    real_words = real_output.split()
    pred_words = pred_output.split()
    
    tp = 0  # True Positives
    fn = 0  # False Negatives

    for real_word, pred_word in zip(real_words, pred_words):
        if real_word.isupper() and pred_word.isupper():
            tp += 1
        elif real_word.isupper() and pred_word.islower():
            fn += 1

    # Recall = TP / (TP + FN)
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    return recall


def metrics_by_file(file_path):

    # Load JSON data from file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Initialize the lists
    accuracy = []
    precision = []
    recall = []

    # Iterate through the results
    for record in data:
        real, pred = record["real_output"], record["pred_output"]
        accuracy_ = calculate_average_accuracy(real, pred)
        precision_ = calculate_average_precision(real, pred)
        recall_ = calculate_average_recall(real, pred)
        accuracy.append(accuracy_)
        precision.append(precision_)
        recall.append(recall_)
    
    average_accuracy = np.mean(accuracy)
    average_precision = np.mean(precision)
    average_recall = np.mean(recall)

    return {
        "average_accuracy": average_accuracy,
        "average_precision": average_precision,
        "average_recall": average_recall,
        "average_f1": 2 * (average_precision * average_recall) / (average_precision + average_recall)
    }

def plot_results(results):

    for metric in ["average_accuracy", "average_precision", "average_recall", "average_f1"]:

        gpt35 = [result[metric] for result in results["gpt-3.5"]]
        print(gpt35)
        gpt4o = [result[metric] for result in results["gpt-4o"]]
        print(gpt4o)

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot the variables
        plt.plot(gpt35[:10], label='GPT-3.5 Turbo FT', color="#001d59", marker='o')
        plt.plot(gpt4o[:10], label='GPT-4o Mini FT', color="#fd8021", marker='o')

        # Plot the horizontal constants
        plt.axhline(y=gpt35[-1], color="#001d59", linestyle='--', label='GPT-3.5 Turbo')
        plt.axhline(y=gpt4o[-1], color="#fd8021", linestyle='--', label='GPT-4o Mini')

        metric_dict = {
            "average_accuracy": "Average Accuracy (%)",
            "average_precision": "Average Precision (%)",
            "average_recall": "Average Recall (%)",
            "average_f1": "Average F1 Score"
        }

        # Add labels and title
        plt.xlabel('Iteration')
        plt.ylabel(metric_dict[metric])

        # Add legend with custom properties
        plt.legend(
            loc='lower right',
            prop={'size': 10},  # Increase the font size
            borderpad=1.2,      # Add padding
            framealpha=1        # Set opacity to 100%
        )

        # Show the plot
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    results_path = "results/GPT/results"
    predictions_path = "results/GPT/predictions"

    # Create a dictionary to store the results
    results = {}
    for model_type in sorted(os.listdir(predictions_path)):

        results[model_type] = []
        for model in sorted(os.listdir(f"{predictions_path}/{model_type}")):
            result = metrics_by_file(f"{predictions_path}/{model_type}/{model}")
            results[model_type].append(result)

    # Plot the results
    plot_results(results)

    
