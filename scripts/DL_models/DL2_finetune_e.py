import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
from matplotlib.backends.backend_pdf import PdfPages

# Suppress warnings
warnings.filterwarnings("ignore")

def plot_accuracy_evolution(trials, model):
    """ Plot the evolution of accuracy over trials """
    trial_numbers = [trial.number for trial in trials]
    accuracies = [trial.values[0] for trial in trials]
    
    plt.figure(figsize=(12, 6))
    plt.plot(trial_numbers, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Trial Number')
    plt.ylabel('Accuracy')
    plt.title('Evolution of Accuracy Over Trials')
    plt.grid(True)
    plt.savefig(f"results/DL/finetune/{model}/plots/accuracy_evolution.png")
    plt.close()

def plot_parameter_importances(study, model):
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
    plt.savefig(f"results/DL/finetune/{model}/plots/parameter_importances.png")
    plt.close()

def plot_parameters_distribution(study, model):
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
        plt.savefig(f"results/DL/finetune/{model}/plots/{idx}_{param}_distribution.png")
        plt.close()

def plot_all_charts_to_png(study, model):
    """ Generate all plots and save them into a single high-quality PNG for each model """
    fig = plt.figure(figsize=(20, 25), dpi=300)  # Increased DPI for higher quality
    
    # Plot accuracy evolution
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    trial_numbers = [trial.number for trial in study.trials]
    accuracies = [trial.values[0] for trial in study.trials]
    
    ax1.plot(trial_numbers, accuracies, marker='o', linestyle='-', color='b')
    ax1.set_xlabel('Trial Number', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_title('Evolution of Accuracy Over Trials', fontsize=16.8)
    ax1.tick_params(axis='both', which='major', labelsize=12.6)
    ax1.grid(True)

    # Plot parameter importances
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    trials_df = pd.DataFrame([trial.params for trial in study.trials])
    trials_df['value'] = [trial.values[0] for trial in study.trials]
    correlations = trials_df.corr()['value'].abs().sort_values(ascending=False)[1:]
    
    sns.barplot(y=correlations.index, x=correlations.values, palette='viridis', ax=ax2)
    ax2.set_ylabel('Parameters', fontsize=14)
    ax2.set_xlabel('Importance (Absolute Correlation)', fontsize=14)
    ax2.set_title('Parameter Importances', fontsize=16.8)
    ax2.tick_params(axis='both', which='major', labelsize=12.6)

    # Plot parameters distribution
    params = trials_df.columns[:-1]  # Exclude 'value' column
    for idx, param in enumerate(params[:4]):  # Plot only the first 4 parameters
        row = idx // 2 + 1
        col = idx % 2
        ax = plt.subplot2grid((3, 2), (row, col))
        
        bins = np.linspace(trials_df[param].min(), trials_df[param].max(), 30)
        binned = pd.cut(trials_df[param], bins=bins, include_lowest=True)
        bin_means = trials_df.groupby(binned)['value'].mean()
        
        norm = Normalize(vmin=bin_means.min(), vmax=bin_means.max())
        cmap = plt.get_cmap('coolwarm')
        sm = ScalarMappable(cmap=cmap, norm=norm)
        
        hist_data, bin_edges = np.histogram(trials_df[param], bins=bins)
        for i in range(len(bin_edges) - 1):
            color = cmap(norm(bin_means.loc[binned.cat.categories[i]]))
            ax.bar(bin_edges[i], hist_data[i], width=bin_edges[i+1] - bin_edges[i],
                    color=color, edgecolor='black')

        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Average Accuracy', fontsize=12.6)
        cbar.ax.tick_params(labelsize=11.2)
        ax.set_xlabel(param, fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title(f'Distribution of {param}', fontsize=16.8)
        ax.tick_params(axis='both', which='major', labelsize=12.6)

    plt.tight_layout()
    plt.savefig(f"results/DL/finetune/{model}/plots/all_charts.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

finetune_path = "results/DL/finetune"

for model in os.listdir(finetune_path):

    try:
        if model == ".DS_Store":
            continue

        if model == "GroNLP":
            model = "GroNLP/hateBERT"
        study = joblib.load(f"{finetune_path}/{model}/study.pkl")
        os.makedirs(f"results/DL/finetune/{model}/plots", exist_ok=True)

        # Print model name, best F1-score, and params
        best_trial = study.best_trial
        print(f"Model: {model}")
        print(f"Best Accuracy: {best_trial.values[0]:.4f}")
        print(f"Best params: {best_trial.params}")
        print("-" * 50)

        # Plot all charts to a single PDF
        plot_all_charts_to_png(study, model)

    except FileNotFoundError:
        print(f"Study not found for model: {model}")
        print("-" * 50)