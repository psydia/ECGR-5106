import torch
import torch.nn as nn

class CIFAR10MLP(nn.Module):
    def __init__(self, hidden_dims=[1024, 512, 256]):
        super(CIFAR10MLP, self).__init__()
        
        # Input dimension for CIFAR-10: 32x32x3 = 3072
        input_dim = 3072
        # Output dimension (number of classes): 10
        output_dim = 10
        
        # First hidden layer
        layers = [
            nn.Flatten(),  # Flatten the 32x32x3 input to 3072
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.3)
        ]
        
        # Add middle hidden layers
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.Dropout(0.3)
            ])
        
        # Output layer
        layers.extend([
            nn.Linear(hidden_dims[-1], output_dim)
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Example usage:
if __name__ == "__main__":
    # Create model instance
    model = CIFAR10MLP(hidden_dims=[1024, 512, 256])
    
    # Test with random input
    batch_size = 32
    test_input = torch.randn(batch_size, 3, 32, 32)  # Example input shape
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [batch_size, 10]