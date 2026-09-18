# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time

from model import create_model

def load_processed_data():
    """Load preprocessed data"""
    print("Loading processed data...")
    data_dir = Path("data")
    data = torch.load(data_dir / "processed_data.pt", weights_only=False)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    num_features = data['num_features']
    
    print(f"✓ Train: {X_train.shape[0]} samples")
    print(f"✓ Test: {X_test.shape[0]} samples")
    print(f"✓ Features: {num_features}")
    
    return X_train, y_train, X_test, y_test, num_features

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return accuracy, precision, recall, f1, all_labels, all_preds

def train_model(epochs=50, batch_size=64, learning_rate=0.001):
    """Main training function"""
    print("="*60)
    print("TRAINING BOT DETECTION MODEL")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    X_train, y_train, X_test, y_test, num_features = load_processed_data()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"\nCreating model...")
    model = create_model(input_dim=num_features).to(device)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    best_f1 = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            accuracy, precision, recall, f1, _, _ = evaluate(model, test_loader, device)
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Acc: {accuracy:.4f} | Prec: {precision:.4f} | "
                  f"Rec: {recall:.4f} | F1: {f1:.4f}")
            
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                model_path = Path("models") / "bot_detector_mlp.pt"
                torch.save(model.state_dict(), model_path)
                print(f"  → Saved best model (F1: {f1:.4f})")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    model.load_state_dict(torch.load(Path("models") / "bot_detector_mlp.pt", weights_only=True))
    accuracy, precision, recall, f1, y_true, y_pred = evaluate(model, test_loader, device)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Human', 'Bot']))
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Model saved to: models/bot_detector_mlp.pt")
    print("="*60)

if __name__ == "__main__":
    train_model(epochs=100, batch_size=32, learning_rate=0.0005)
