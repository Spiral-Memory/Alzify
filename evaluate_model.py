import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from model import build_model
import pandas as pd
import multiprocessing
import random
import numpy as np

def calculate_metrics(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, output_dict=True)

    return cm, report

def display_metrics_table(report):
    metrics_table = pd.DataFrame(report).transpose()
    if 'accuracy' in metrics_table.columns:
        metrics_table = metrics_table.drop(columns=['accuracy', 'macro avg', 'weighted avg'])
    metrics_table['Class'] = metrics_table.index
    metrics_table = metrics_table[['Class', 'precision', 'recall']]
    metrics_table = metrics_table.round(4)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(metrics_table.set_index('Class'), annot=True, cmap='Blues', cbar=False)
    plt.title('Classification Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig('metrics_plot.png')  # Save the plot as an image
    plt.show()



def plot_confusion_matrix(true_labels, predicted_labels, class_labels):
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()


def main():

    torch.manual_seed(42)


    IMG_SIZE = 128
    BRIGHT_RANGE = [0.8, 1.2]
    transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=BRIGHT_RANGE, contrast=BRIGHT_RANGE, saturation=BRIGHT_RANGE, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

    batch_size = 32

    # Load the dataset
    train_dataset = datasets.ImageFolder('Alzheimer/train', transform=transform)
    test_dataset = datasets.ImageFolder('Alzheimer/test', transform=transform)

    total_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    print(f"Total dataset size: {len(total_dataset)}")

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(total_dataset))
    test_size = len(total_dataset) - train_size
    train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")


    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n")

    model = build_model(pretrained=False, fine_tune=False).to(device)
    print('[INFO]: Loading custom-trained weights...')
    checkpoint = torch.load('outputs/model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    predicted_labels = []
    true_labels = []
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # print(f"Output : " ,outputs)

            # probab = torch.nn.functional.softmax(outputs, dim=1)
            # print(f"Probability : ", probab)


            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            correct_predictions += (predicted == labels).sum().item()

    test_accuracy = correct_predictions / len(test_dataset) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Compute metrics
    cm, report = calculate_metrics(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)
    print("\n")
    display_metrics_table(report)

    # Plot the confusion matrix
    class_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
    plot_confusion_matrix(true_labels, predicted_labels, class_labels)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
