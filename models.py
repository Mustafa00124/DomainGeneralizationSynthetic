import torch
import torch.nn as nn
import os
from constants import *

class DomainGeneralizationModel(nn.Module):
    def __init__(self):
        super(DomainGeneralizationModel, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN1_SIZE)
        self.fc2 = nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE)
        self.classifier = nn.Linear(HIDDEN2_SIZE, OUTPUT_SIZE)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.classifier(x))
        return x
    
    def get_latent_representation(self, x):
        """Get latent representation (output of second hidden layer)"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

def save_model(model, epoch, method, save_dir=None):
    """Save model checkpoint"""
    if save_dir is None:
        save_dir = os.path.join(MODELS_DIR, EPOCHS_DIR)
    
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'method': method,
        'model_config': {
            'input_size': INPUT_SIZE,
            'hidden1_size': HIDDEN1_SIZE,
            'hidden2_size': HIDDEN2_SIZE,
            'output_size': OUTPUT_SIZE
        }
    }
    
    filename = f"{save_dir}/model_{method}_epoch_{epoch:04d}.pth"
    torch.save(checkpoint, filename)
    print(f"Model saved: {filename}")
    return filename

def load_model(filename, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(filename, map_location=device)
    
    model = DomainGeneralizationModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def list_saved_models(epochs_dir=None):
    """List all saved model checkpoints"""
    if epochs_dir is None:
        epochs_dir = os.path.join(MODELS_DIR, EPOCHS_DIR)
    
    if not os.path.exists(epochs_dir):
        print(f"No models directory found at {epochs_dir}")
        return []
    
    models = []
    for filename in os.listdir(epochs_dir):
        if filename.endswith('.pth'):
            models.append(os.path.join(epochs_dir, filename))
    
    models.sort()
    return models

if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    model = DomainGeneralizationModel()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(5, INPUT_SIZE)
    output = model(x)
    latent = model.get_latent_representation(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Latent shape: {latent.shape}")
    
    # Test model saving/loading
    print("\nTesting model saving/loading...")
    save_model(model, 1, "test")
    
    loaded_model, checkpoint = load_model("models/epochs/model_test_epoch_0001.pth")
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Clean up test file
    os.remove("models/epochs/model_test_epoch_0001.pth")
    print("Test complete!")
