# src/feature_extraction.py
import torch
import numpy as np
from pathlib import Path

def load_mgtab_data():
    """Load MGTAB tensor data"""
    print("Loading MGTAB data...")
    
    data_dir = Path("data")
    
    # Load features and labels
    features = torch.load(data_dir / "features.pt")
    labels = torch.load(data_dir / "labels_bot.pt")
    
    print(f"✓ Loaded features: {features.shape}")
    print(f"✓ Loaded labels: {labels.shape}")
    
    return features, labels

def extract_raw_features(features):
    """
    Extract the 20 most important raw features from MGTAB
    MGTAB features.pt contains user property features
    """
    print("\nExtracting 20 raw features...")
    
    # MGTAB typically has features in this order (first 20 columns)
    # These are the most important based on information gain
    if features.shape[1] >= 20:
        raw_features = features[:, :20]
    else:
        raw_features = features
    
    print(f"✓ Extracted features shape: {raw_features.shape}")
    return raw_features

def add_derived_features(features):
    """
    Add derived features to improve bot detection
    Assumes features has at least: followers, following, posts
    """
    print("\nAdding derived features...")
    
    # Extract key features (adjust indices based on actual MGTAB structure)
    # Typical MGTAB structure: [followers, following, statuses, ...]
    followers = features[:, 0] if features.shape[1] > 0 else torch.ones(features.shape[0])
    following = features[:, 1] if features.shape[1] > 1 else torch.ones(features.shape[0])
    posts = features[:, 2] if features.shape[1] > 2 else torch.ones(features.shape[0])
    
    # Derived feature 1: Follower-to-Following Ratio (most important)
    follower_ratio = followers / (following + 1e-6)
    
    # Derived feature 2: Following-to-Follower Ratio
    following_ratio = following / (followers + 1e-6)
    
    # Derived feature 3: Bot Score Composite
    # Combines ratio extremes, posting frequency signals
    ratio_score = torch.clamp(torch.abs(torch.log10(follower_ratio + 1e-6)), 0, 3)
    
    # Stack derived features
    derived = torch.stack([
        follower_ratio,
        following_ratio,
        ratio_score
    ], dim=1)
    
    # Concatenate with original features
    features_extended = torch.cat([features, derived], dim=1)
    
    print(f"✓ Added 3 derived features")
    print(f"✓ Final feature shape: {features_extended.shape}")
    
    return features_extended

def prepare_dataset(test_size=0.2, random_seed=42):
    """
    Prepare complete dataset with train/test split
    """
    print("="*50)
    print("PREPARING MGTAB DATASET")
    print("="*50)
    
    # Load data
    features, labels = load_mgtab_data()
    
    # Extract raw features
    raw_features = extract_raw_features(features)
    
    # Add derived features
    final_features = add_derived_features(raw_features)
    
    # Train/test split
    print(f"\nSplitting data (test_size={test_size})...")
    num_samples = final_features.shape[0]
    num_test = int(num_samples * test_size)
    
    # Shuffle indices
    torch.manual_seed(random_seed)
    indices = torch.randperm(num_samples)
    
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    
    # Split data
    X_train = final_features[train_indices]
    y_train = labels[train_indices]
    X_test = final_features[test_indices]
    y_test = labels[test_indices]
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Feature dimension: {X_train.shape[1]}")
    
    # Save processed data
    data_dir = Path("data")
    torch.save({
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'num_features': X_train.shape[1]
    }, data_dir / "processed_data.pt")
    
    print(f"\n✅ Processed data saved to {data_dir / 'processed_data.pt'}")
    print("="*50)
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    prepare_dataset()
