# src/model.py
import torch
import torch.nn as nn

class BotDetectorMLP(nn.Module):
    """
    Multi-Layer Perceptron for Bot Detection
    Simple 3-layer architecture optimized for 23 input features
    """
    def __init__(self, input_dim=23, hidden_dim1=128, hidden_dim2=64, output_dim=2, dropout=0.3):
        super(BotDetectorMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(hidden_dim2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        """
        Convenience method for inference
        Returns: predicted class (0=human, 1=bot)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

def create_model(input_dim=23):
    """Factory function to create model"""
    model = BotDetectorMLP(input_dim=input_dim)
    return model

if __name__ == "__main__":
    # Test model creation
    print("Testing model architecture...")
    model = create_model(input_dim=23)
    print(f"✓ Model created successfully")
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(32, 23)  # Batch of 32 samples, 23 features
    output = model(dummy_input)
    print(f"\n✓ Forward pass successful")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Total parameters: {total_params:,}")
