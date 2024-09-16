import os
import json
import matplotlib.pyplot as plt

# Plot training and evaluation metrics for each model
def plot_loss(model_names):
    colors = {'train': '#001d59', 'eval': '#fd8021'}
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    for i, model_name in enumerate(model_names):
        metrics_path = f'results/DL/optimized/{model_name}/'

        with open(os.path.join(metrics_path, 'eval_metrics.json')) as f:
            eval_metrics = json.load(f)

        with open(os.path.join(metrics_path, 'train_metrics.json')) as f:
            train_metrics = json.load(f)

        epochs = range(1, len(eval_metrics) + 1)
        eval_metric_values = [entry["eval_loss"] for entry in eval_metrics]
        train_metric_values = [entry["loss"] for entry in train_metrics]

        ax = axs[i // 2, i % 2]
        ax.plot(epochs, train_metric_values, label='Training', color=colors['train'])
        ax.plot(epochs, eval_metric_values, label='Evaluation', color=colors['eval'], linestyle='--')
        ax.set_ylabel(model_name)
        ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('results/DL/optimized/plots/loss_plot.png')
    plt.close()


# Plot metrics for all models
def plot_metrics(model_names):
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    colors = ['#001d59', '#fd8021', '#4a86e8', '#2ca02c']
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        
        for j, model_name in enumerate(model_names):
            metrics_path = f'results/DL/optimized/{model_name}/'
            eval_metrics_file = os.path.join(metrics_path, 'eval_metrics.json')

            with open(eval_metrics_file) as f:
                eval_metrics = json.load(f)

            epochs = range(1, len(eval_metrics) + 1)
            metric_values = [entry[f'eval_{metric}'] for entry in eval_metrics]
            ax.plot(epochs, metric_values, label=model_name, color=colors[j])

        ax.grid(True)
        ax.set_ylabel(metric.capitalize())
        ax.legend(
            loc='lower right',
            prop={'size': 10},
            borderpad=1.2,
            framealpha=1
        )

    plt.tight_layout()
    plt.savefig('results/DL/optimized/plots/combined_metrics_plot.png')
    plt.close()

# List of BERT models
bert_models = ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'hatebert']

plot_loss(bert_models)
plot_metrics(bert_models)
