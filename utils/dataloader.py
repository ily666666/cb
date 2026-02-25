import torch
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as scio
import numpy as np
import random
from .readdata_25 import subDataset


def get_dataloaders(data_path, batch_size=32, num_workers=4, allowed_classes=None):
    """
    Create data loaders for training, validation and testing
    
    Args:
        data_path: Path to the dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        allowed_classes: Optional list/set of class ids to include. If None, include all classes.
    
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
        num_classes: Number of classes in the dataset
    """
    
    # Create datasets
    train_dataset = subDataset(datapath=data_path, split='train', transform=None, allowed_classes=allowed_classes)
    val_dataset = subDataset(datapath=data_path, split='valid', transform=None, allowed_classes=allowed_classes)
    test_dataset = subDataset(datapath=data_path, split='test', transform=None, allowed_classes=allowed_classes)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # Get number of classes from the training dataset
    num_classes = train_dataset.num_classes
    
    return train_loader, val_loader, test_loader, num_classes


if __name__ == '__main__':
    # Test the dataloaders
    data_path = 'E\\LoRa_Outdoor_Dataset_Day5\\'
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(data_path, batch_size=4)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Check one batch
    for data, labels in train_loader:
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        break 