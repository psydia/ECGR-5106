import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class CIFAR10Datasets:
    def __init__(self, root_dir, val_split=0.1, batch_size=32):
        # Define basic transformations
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load training data
        full_train_dataset = datasets.CIFAR10(
            root=root_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        # Calculate split sizes
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        
        # Split into train and validation
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, 
            [train_size, val_size]
        )
        
        # Load test data
        self.test_dataset = datasets.CIFAR10(
            root=root_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Save dataset info
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        self.batch_size = batch_size
        
    def get_dataset_sizes(self):
        return {
            'train': len(self.train_dataset),
            'validation': len(self.val_dataset),
            'test': len(self.test_dataset)
        }
    
    def get_loaders(self):
        return {
            'train': self.train_loader,
            'validation': self.val_loader,
            'test': self.test_loader
        }

# Example usage
if __name__ == "__main__":
    # Initialize datasets and loaders
    cifar_data = CIFAR10Datasets(
        root_dir='./data',
        val_split=0.1,  # 10% validation split
        batch_size=32
    )
    
    # Get dataset sizes
    sizes = cifar_data.get_dataset_sizes()
    print("Dataset sizes:")
    for split, size in sizes.items():
        print(f"{split}: {size}")
    
    # Get data loaders
    loaders = cifar_data.get_loaders()
    
    # Example of iterating through training data
    for images, labels in loaders['train']:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break  # Just print first batch
        
    # Access class names
    print("\nAvailable classes:", cifar_data.classes)