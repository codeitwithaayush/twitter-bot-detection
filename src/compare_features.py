import torch
from inference import BotDetector

# Load training features (from processed_data.pt)
train_data = torch.load('data/processed_data.pt')
# Take the first training sample
sample_train = train_data['X_train'][0]
print("Training sample (first 10):", sample_train[:10])
print("Training sample (last 3):", sample_train[-3:])

# Initialize bot detector (loads normalization and model)
detector = BotDetector()

# Use a dummy user for feature extraction (bypasses API)
features = detector._create_dummy_features()
print("Extracted features (first 10):", features[:10])
print("Extracted features (last 3):", features[-3:])
