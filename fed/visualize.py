import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_training_progress(train_losses, train_accs, val_losses, val_accs, save_path):
    """
    Plot training and validation losses and accuracies
    
    Args:
        train_losses: List of training losses
        train_accs: List of training accuracies
        val_losses: List of validation losses
        val_accs: List of validation accuracies
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def extract_features(model, data_loader, device, layer_name=None):
    """
    Extract features from a specific layer or the penultimate layer of the model
    
    Args:
        model: The model
        data_loader: DataLoader containing the data
        device: Device to use for extraction (cpu or cuda)
        layer_name: Name of the layer to extract features from (if None, use the layer before FC)
        
    Returns:
        features: Extracted features
        labels: Corresponding labels
    """
    model.eval()
    features = []
    all_labels = []
    
    # Define hook for feature extraction
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # For complex output, convert to real by concatenating real and imaginary parts
            if torch.is_complex(output):
                real_part = output.real
                imag_part = output.imag
                # Flatten and concatenate
                batch_size = real_part.size(0)
                real_flat = real_part.view(batch_size, -1)
                imag_flat = imag_part.view(batch_size, -1)
                activation[name] = torch.cat([real_flat, imag_flat], dim=1).detach()
            else:
                activation[name] = output.detach()
        return hook
    
    # Register hook
    if layer_name:
        # For specific layer
        if hasattr(model, layer_name):
            hook_handle = getattr(model, layer_name).register_forward_hook(get_activation(layer_name))
        else:
            raise ValueError(f"Layer {layer_name} not found in model")
    else:
        # For penultimate layer (before FC)
        hook_handle = model.avgpool.register_forward_hook(get_activation('avgpool'))
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            model(data)  # Forward pass
            
            # Get features from the layer
            if layer_name:
                batch_features = activation[layer_name].cpu().numpy()
            else:
                batch_features = activation['avgpool'].cpu().numpy()
                
            features.append(batch_features)
            all_labels.append(labels.cpu().numpy())
    
    # Remove hook
    hook_handle.remove()
    
    # Concatenate all features and labels
    features = np.concatenate(features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return features, all_labels


def visualize_features_tsne(features, labels, class_names, save_path, perplexity=30, n_iter=1000):
    """
    Visualize features using t-SNE
    
    Args:
        features: Extracted features
        labels: Corresponding labels
        class_names: List of class names
        save_path: Path to save the plot
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
    """
    # Apply t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    
    # If features are too large, sample a subset
    max_samples = 5000
    if features.shape[0] > max_samples:
        indices = np.random.choice(features.shape[0], max_samples, replace=False)
        features_subset = features[indices]
        labels_subset = labels[indices]
    else:
        features_subset = features
        labels_subset = labels
    
    features_tsne = tsne.fit_transform(features_subset)
    
    # Plot t-SNE
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels_subset, 
               cmap='tab20', alpha=0.7, s=40)
    
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                         title="Classes")
    plt.gca().add_artist(legend1)
    
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_features_pca(features, labels, class_names, save_path):
    """
    Visualize features using PCA
    
    Args:
        features: Extracted features
        labels: Corresponding labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Apply PCA
    print("Computing PCA embedding...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # Plot PCA
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, 
               cmap='tab20', alpha=0.7, s=40)
    
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                         title="Classes")
    plt.gca().add_artist(legend1)
    
    plt.title('PCA Visualization of Features')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_model_summary(model, input_shape, save_path):
    """
    Visualize model summary statistics
    
    Args:
        model: The model
        input_shape: Shape of input data
        save_path: Path to save the summary
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create a text file with model statistics
    with open(save_path, 'w') as f:
        f.write(f"Model Architecture: {model.__class__.__name__}\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n")
        f.write("\nModel Structure:\n")
        f.write(str(model))


def visualize_complex_signal(data, sample_idx=0, class_name="Unknown", save_path=None):
    """
    Visualize complex signal
    
    Args:
        data: Complex data tensor
        sample_idx: Index of sample to visualize
        class_name: Name of the class
        save_path: Path to save the plot (if None, just display)
    """
    # Get single sample
    signal = data[sample_idx].cpu().numpy()
    
    plt.figure(figsize=(15, 10))
    
    # Plot real part
    plt.subplot(2, 2, 1)
    plt.plot(np.real(signal))
    plt.title(f"Real Part - Class: {class_name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot imaginary part
    plt.subplot(2, 2, 2)
    plt.plot(np.imag(signal))
    plt.title(f"Imaginary Part - Class: {class_name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot magnitude
    plt.subplot(2, 2, 3)
    plt.plot(np.abs(signal))
    plt.title("Magnitude")
    plt.xlabel("Sample")
    plt.ylabel("Magnitude")
    plt.grid(True)
    
    # Plot phase
    plt.subplot(2, 2, 4)
    plt.plot(np.angle(signal))
    plt.title("Phase")
    plt.xlabel("Sample")
    plt.ylabel("Phase (rad)")
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 