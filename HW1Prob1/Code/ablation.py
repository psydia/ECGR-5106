import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from data_loader import CIFAR10Datasets
from model import CIFAR10MLP

def train_model(model, data_loaders, criterion, optimizer, model_name, num_epochs=20, device='cuda', checkpoint_dir='checkpoints'):
    # Create checkpoint directory if it doesn't exist
    model_checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize tracking variables
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(data_loaders['train'], desc=f'{model_name} - Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(data_loaders['train'])
        epoch_train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loaders['validation']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(data_loaders['validation'])
        epoch_val_acc = 100 * val_correct / val_total
        
        # Store metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Save checkpoint if validation loss improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_acc': epoch_train_acc,
                'val_acc': epoch_val_acc,
                'history': history
            }
            checkpoint_path = os.path.join(model_checkpoint_dir, 'best_model_checkpoint.pth')
            torch.save(checkpoint, checkpoint_path)
            
            # Save history to JSON for easy loading
            history_path = os.path.join(model_checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f)
            
            print(f'{model_name} - Checkpoint saved! Best validation loss: {best_val_loss:.4f}')
        
        # Print epoch statistics
        print(f'{model_name} - Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print('-' * 60)
    
    return history, best_val_loss

def plot_comparison(histories, title):
    """Plot training histories for multiple models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    for model_name, history in histories.items():
        ax1.plot(history['train_loss'], label=f'{model_name} - Train')
        ax1.plot(history['val_loss'], '--', label=f'{model_name} - Val')
    
    ax1.set_title('Model Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training and validation accuracy
    for model_name, history in histories.items():
        ax2.plot(history['train_acc'], label=f'{model_name} - Train')
        ax2.plot(history['val_acc'], '--', label=f'{model_name} - Val')
    
    ax2.set_title('Model Accuracy Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize dataset
    cifar_data = CIFAR10Datasets(root_dir='./data', val_split=0.1, batch_size=128)
    data_loaders = cifar_data.get_loaders()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model architectures
    architectures = {
        'base_model': [1024, 512, 256],
        'wide_model': [2048, 1024, 512, 256],
        'deep_model': [1024, 1024, 512, 512, 256, 256]
    }
    
    # Training settings
    num_epochs = 20
    criterion = nn.CrossEntropyLoss()
    histories = {}
    
    # Train each model
    for model_name, hidden_dims in architectures.items():
        print(f"\nTraining {model_name} with architecture: {hidden_dims}")
        
        # Initialize model and optimizer
        model = CIFAR10MLP(hidden_dims=hidden_dims)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        history, best_val_loss = train_model(
            model=model,
            data_loaders=data_loaders,
            criterion=criterion,
            optimizer=optimizer,
            model_name=model_name,
            num_epochs=num_epochs,
            device=device
        )
        
        histories[model_name] = history
        print(f"\n{model_name} completed training. Best validation loss: {best_val_loss:.4f}")
    
    # Plot comparison
    plot_comparison(histories, 'Model Architecture Comparison')

if __name__ == "__main__":
    main()