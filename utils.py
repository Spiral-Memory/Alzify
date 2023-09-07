import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report

matplotlib.style.use('ggplot')


def save_model(epochs, model, optimizer, criterion):
    state = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion
    }
    torch.save(state, 'outputs/model.pth')


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    
    plt.figure(figsize=(10, 7))

    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy')
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='valid accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')

    # loss plot

    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')


def plot_confusion_matrix(true_labels, predicted_labels, class_labels):
    
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig('outputs/cmatrix.png')


def calculate_metrics(true_labels, predicted_labels):   
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    return report

def display_metrics_table(report):    
    metrics_table = pd.DataFrame(report).transpose()
    if 'accuracy' in metrics_table.columns:
        metrics_table = metrics_table.drop(columns=['accuracy', 'macro avg', 'weighted avg'])
    metrics_table['Class'] = metrics_table.index
    metrics_table = metrics_table[['Class', 'Precision', 'Recall']]
    metrics_table = metrics_table.round(4)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(metrics_table.set_index('Class'), annot=True, cmap='Blues', cbar=False)
    plt.title('Classification Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig('outputs/metrics_plot.png')  # Save the plot as an image
    plt.show()




