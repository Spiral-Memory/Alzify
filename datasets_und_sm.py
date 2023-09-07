import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from imblearn.over_sampling import SMOTE
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

# Define image transformations including data augmentation

torch.manual_seed(42)

IMG_SIZE = 128
BRIGHT_RANGE = [0.8, 1.2]
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(brightness=BRIGHT_RANGE, contrast=BRIGHT_RANGE, saturation=BRIGHT_RANGE, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

def create_data_loaders():
    # Load the dataset
    train_dataset = ImageFolder('Alzheimer/train', transform=transform)
    test_dataset = ImageFolder('Alzheimer/test', transform=transform)

    total_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    print(f"Total dataset size: {len(total_dataset)}")

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(total_dataset))
    test_size = len(total_dataset) - train_size
    train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create train and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=6400, shuffle=True)

    train_data, train_labels = next(iter(train_loader))

    print(train_data.shape) 

    train_data = train_data.numpy().reshape(-1, IMG_SIZE * IMG_SIZE * 3)

    print(train_data.shape)  

    rus = RandomUnderSampler(sampling_strategy= {0: 685, 1: 50, 2: 2200, 3: 1789})
    train_data, train_labels = rus.fit_resample(train_data, train_labels)

    unique_labels, label_counts = np.unique(train_labels, return_counts=True)

    # Print the unique labels and their counts
    for label, count in zip(unique_labels, label_counts):
        print(f"Label: {label}, Count: {count}")

    print(5 * "-")

    # Apply SMOTE oversampling to minority classes
    sm = SMOTE(random_state=42)
    train_data, train_labels = sm.fit_resample(train_data, train_labels)

    unique_labels, label_counts = np.unique(train_labels, return_counts=True)

    # Print the unique labels and their counts
    for label, count in zip(unique_labels, label_counts):
        print(f"Label: {label}, Count: {count}")

    print(5 * "-")


    # Reshape train data to PyTorch tensor format
    train_data = torch.from_numpy(train_data.reshape(-1, 3, IMG_SIZE, IMG_SIZE))
    train_labels = torch.from_numpy(train_labels)

    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")

    unique_labels, label_counts = np.unique(train_labels, return_counts=True)

    # Print the unique labels and their counts
    for label, count in zip(unique_labels, label_counts):
        print(f"Label: {label}, Count: {count}")

    oversampled_train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    return oversampled_train_dataset, test_dataset



if __name__ == '__main__':
    create_data_loaders()