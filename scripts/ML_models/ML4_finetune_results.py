import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

study_file = "results/ML/finetune/LGBM_tfidf_10000_2.pkl"

def plot_accuracy_evolution(trials):
    """ Plot the evolution of accuracy over trials """
    trial_numbers = [trial.number for trial in trials]
    accuracies = [trial.values[0] for trial in trials]
    
    plt.figure(figsize=(12, 6))
    plt.plot(trial_numbers, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Trial Number')
    plt.ylabel('Accuracy')
    plt.title('Evolution of Accuracy Over Trials')
    plt.grid(True)
    plt.savefig("results/ML/finetune/best_model/accuracy_evolution.png")
    plt.close()

def plot_parameter_importances(study):
    """ Plot the parameter importances as a bar chart """
    # Convert study's trials into a DataFrame
    trials_df = pd.DataFrame([trial.params for trial in study.trials])
    trials_df['value'] = [trial.values[0] for trial in study.trials]
    
    # Calculate parameter importances
    correlations = trials_df.corr()['value'].abs().sort_values(ascending=False)[1:]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=correlations.index, y=correlations.values, palette='viridis')
    plt.xlabel('Parameters')
    plt.ylabel('Importance (Absolute Correlation)')
    plt.title('Parameter Importances')
    plt.xticks(rotation=90)
    plt.savefig("results/ML/finetune/best_model/parameter_importances.png")
    plt.close()

def plot_parameters_distribution(study):
    """ Plot the distribution of all parameters with color-coded bins showing average accuracy """
    # Convert study's trials into a DataFrame
    trials_df = pd.DataFrame([trial.params for trial in study.trials])
    trials_df['value'] = [trial.values[0] for trial in study.trials]
    
    for idx, param in enumerate(trials_df.columns[:-1]):  # Exclude 'value' column
        plt.figure(figsize=(14, 7))
        
        # Create bins and calculate average accuracy in each bin
        bins = np.linspace(trials_df[param].min(), trials_df[param].max(), 30)
        binned = pd.cut(trials_df[param], bins=bins, include_lowest=True)
        bin_means = trials_df.groupby(binned)['value'].mean()
        bin_counts = trials_df.groupby(binned)[param].count()
        
        # Create the plot for the parameter distribution with color-coded bins
        ax1 = plt.subplot(1, 1, 1)
        norm = Normalize(vmin=bin_means.min(), vmax=bin_means.max())
        cmap = plt.get_cmap('coolwarm')
        sm = ScalarMappable(cmap=cmap, norm=norm)
        
        # Plot histogram
        hist_data, bin_edges = np.histogram(trials_df[param], bins=bins)
        for i in range(len(bin_edges) - 1):
            color = cmap(norm(bin_means.loc[binned.cat.categories[i]]))
            ax1.bar(bin_edges[i], hist_data[i], width=bin_edges[i+1] - bin_edges[i],
                    color=color, edgecolor='black')

        cbar = plt.colorbar(sm, ax=ax1, orientation='vertical')
        cbar.set_label('Average Accuracy')
        plt.xlabel(param)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {param} with Accuracy Coloring')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"results/ML/finetune/best_model/{idx}_{param}_distribution.png")
        plt.close()

try:
    study = joblib.load(study_file)
    print(f"Loaded study: {study_file}")

    # Plot accuracy evolution
    plot_accuracy_evolution(study.trials)

    # Plot parameter importances
    plot_parameter_importances(study)

    # Plot distribution of the top 3 parameters
    plot_parameters_distribution(study)

except FileNotFoundError:
    print(f"Study {study_file} not found.")