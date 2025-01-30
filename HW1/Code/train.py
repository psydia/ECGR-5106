import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from data_loader import CIFAR10Datasets
from model import CIFAR10MLP

def train_model(model, data_loaders, criterion, optimizer, num_epochs=20, device='cuda', checkpoint_dir='checkpoints'):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
        
        for inputs, labels in tqdm(data_loaders['train'], desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
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
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track statistics
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
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model_checkpoint.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved! Best validation loss: {best_val_loss:.4f}')
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print('-' * 60)
    
    return history, best_val_loss

def plot_training_history(history):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Function to load checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (checkpoint['epoch'], checkpoint['train_loss'], 
            checkpoint['val_loss'], checkpoint['train_acc'], 
            checkpoint['val_acc'], checkpoint['history'])

# Main training script
if __name__ == "__main__":
    # Initialize datasets and model
    cifar_data = CIFAR10Datasets(root_dir='/home/ssiraz/Development/PhD Courses/Real time Machine Learning/HW1/Dataset', val_split=0.1, batch_size=128)
    model = CIFAR10MLP(hidden_dims=[1024, 512, 256])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    history, best_val_loss = train_model(
        model=model,
        data_loaders=cifar_data.get_loaders(),
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        device=device,
        checkpoint_dir='checkpoints'
    )
    
    # Plot training history
    plot_training_history(history)
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    
    # Example of loading the best checkpoint
    best_checkpoint_path = '/home/ssiraz/Development/PhD Courses/Real time Machine Learning/HW1/checkpoints/best_model_checkpoint.pth'
    if os.path.exists(best_checkpoint_path):
        epoch, train_loss, val_loss, train_acc, val_acc, _ = load_checkpoint(
            model, optimizer, best_checkpoint_path)
        print(f"\nLoaded best checkpoint from epoch {epoch}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")