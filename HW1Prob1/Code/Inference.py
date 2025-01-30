import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from data_loader import CIFAR10Datasets
from model import CIFAR10MLP

def evaluate_model(model, data_loader, criterion, device='cuda'):
    """
    Evaluate the model and calculate various performance metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Accumulate statistics
            total_loss += loss.item()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Store predictions and labels for metric calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy and loss
    accuracy = 100 * total_correct / total_samples
    avg_loss = total_loss / len(data_loader)
    
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average=None
    )
    
    # Calculate macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average='macro'
    )
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': conf_matrix
    }

def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plot confusion matrix using seaborn
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def print_metrics(metrics, class_names):
    """
    Print all metrics in a formatted way
    """
    print(f"\nModel Performance Metrics:")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Average Loss: {metrics['loss']:.4f}")
    print(f"\nMacro-averaged Metrics:")
    print(f"{'='*50}")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall: {metrics['macro_recall']:.4f}")
    print(f"F1 Score: {metrics['macro_f1']:.4f}")
    
    print(f"\nPer-class Metrics:")
    print(f"{'='*50}")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    print(f"{'-'*50}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {metrics['precision'][i]:>10.4f} {metrics['recall'][i]:>10.4f} {metrics['f1'][i]:>10.4f}")

if __name__ == "__main__":
    # Initialize datasets and model
    cifar_data = CIFAR10Datasets(
        root_dir='./data',
        val_split=0.1,
        batch_size=128
    )
    
    # Initialize model and load checkpoint
    model = CIFAR10MLP(hidden_dims=[1024, 512, 256])
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load the best checkpoint
    checkpoint_path = 'checkpoints/best_model_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get test data loader
    test_loader = cifar_data.get_loaders()['test']
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, criterion, device)
    
    # Print metrics
    print_metrics(metrics, cifar_data.classes)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], cifar_data.classes)