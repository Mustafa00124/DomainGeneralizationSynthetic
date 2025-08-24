import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
from constants import *
from models import save_model
from utils import get_grads

def train_model_erm(model, train_data, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, 
                   learning_rate=DEFAULT_LEARNING_RATE, device='cpu'):
    """Train model using ERM (Empirical Risk Minimization) with gradient averaging across domains"""
    
    # Convert data to tensors and create data loaders
    X1_train, Y1_train = train_data['domain1']['train']
    X2_train, Y2_train = train_data['domain2']['train']
    
    # Convert labels from [-1, 1] to [0, 1] for binary classification
    Y1_train_binary = (Y1_train + 1) / 2
    Y2_train_binary = (Y2_train + 1) / 2
    
    # Create data loaders
    train_dataset1 = TensorDataset(torch.FloatTensor(X1_train), torch.FloatTensor(Y1_train_binary).unsqueeze(1))
    train_dataset2 = TensorDataset(torch.FloatTensor(X2_train), torch.FloatTensor(Y2_train_binary).unsqueeze(1))
    
    train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    overall_losses = []
    overall_accuracies = []
    domain1_losses = []
    domain1_accuracies = []
    domain2_losses = []
    domain2_accuracies = []
    
    model.train()
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        # Progress bar for this epoch
        pbar = tqdm(total=len(train_loader1), desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, ((X1_batch, Y1_batch), (X2_batch, Y2_batch)) in enumerate(zip(train_loader1, train_loader2)):
            X1_batch, Y1_batch = X1_batch.to(device), Y1_batch.to(device)
            X2_batch, Y2_batch = X2_batch.to(device), Y2_batch.to(device)
            
            # Forward pass for both domains
            optimizer.zero_grad()
            
            # Domain 1
            outputs1 = model(X1_batch)
            loss1 = criterion(outputs1, Y1_batch)
            
            # Domain 2
            outputs2 = model(X2_batch)
            loss2 = criterion(outputs2, Y2_batch)
            
            # Average loss
            total_loss = (loss1 + loss2) / 2
            
            # Backward pass
            total_loss.backward()
            
            # Average gradients across domains
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data /= 2
            
            optimizer.step()
            
            # Calculate accuracy
            pred1 = (outputs1 > 0.5).float()
            pred2 = (outputs2 > 0.5).float()
            acc1 = (pred1 == Y1_batch).float().mean()
            acc2 = (pred2 == Y2_batch).float().mean()
            
            epoch_losses.append(total_loss.item())
            epoch_accuracies.append((acc1 + acc2).item() / 2)
            
            pbar.update(1)
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}', 'Acc': f'{epoch_accuracies[-1]:.4f}'})
        
        pbar.close()
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        
        # Store history
        overall_losses.append(avg_loss)
        overall_accuracies.append(avg_acc)
        
        # Calculate domain-specific metrics
        model.eval()
        with torch.no_grad():
            # Domain 1
            X1_tensor = torch.FloatTensor(X1_train).to(device)
            Y1_tensor = torch.FloatTensor(Y1_train_binary).unsqueeze(1).to(device)
            outputs1 = model(X1_tensor)
            loss1 = criterion(outputs1, Y1_tensor)
            pred1 = (outputs1 > 0.5).float()
            acc1 = (pred1 == Y1_tensor).float().mean()
            
            domain1_losses.append(loss1.item())
            domain1_accuracies.append(acc1.item())
            
            # Domain 2
            X2_tensor = torch.FloatTensor(X2_train).to(device)
            Y2_tensor = torch.FloatTensor(Y2_train_binary).unsqueeze(1).to(device)
            outputs2 = model(X2_tensor)
            loss2 = criterion(outputs2, Y2_tensor)
            pred2 = (outputs2 > 0.5).float()
            acc2 = (pred2 == Y2_tensor).float().mean()
            
            domain2_losses.append(loss2.item())
            domain2_accuracies.append(acc2.item())
        
        model.train()
        
        # Save model every 10 epochs and at the end
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            save_model(model, epoch + 1, "erm")
        
        print(f'Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}')
    
    # Save training history
    training_history = {
        'overall_losses': overall_losses,
        'overall_accuracies': overall_accuracies,
        'domain1_losses': domain1_losses,
        'domain1_accuracies': domain1_accuracies,
        'domain2_losses': domain2_losses,
        'domain2_accuracies': domain2_accuracies
    }
    
    save_training_history(training_history, "erm")
    
    return training_history

def train_model_gradient_aligned(model, train_data, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                               learning_rate=DEFAULT_LEARNING_RATE, device='cpu'):
    """Train model using gradient alignment optimization"""
    
    # Convert data to tensors and create data loaders
    X1_train, Y1_train = train_data['domain1']['train']
    X2_train, Y2_train = train_data['domain2']['train']
    
    # Convert labels from [-1, 1] to [0, 1] for binary classification
    Y1_train_binary = (Y1_train + 1) / 2
    Y2_train_binary = (Y2_train + 1) / 2
    
    # Create data loaders
    train_dataset1 = TensorDataset(torch.FloatTensor(X1_train), torch.FloatTensor(Y1_train_binary).unsqueeze(1))
    train_dataset2 = TensorDataset(torch.FloatTensor(X2_train), torch.FloatTensor(Y2_train_binary).unsqueeze(1))
    
    train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    overall_losses = []
    overall_accuracies = []
    domain1_losses = []
    domain1_accuracies = []
    domain2_losses = []
    domain2_accuracies = []
    
    model.train()
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        # Progress bar for this epoch
        pbar = tqdm(total=len(train_loader1), desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, ((X1_batch, Y1_batch), (X2_batch, Y2_batch)) in enumerate(zip(train_loader1, train_loader2)):
            X1_batch, Y1_batch = X1_batch.to(device), Y1_batch.to(device)
            X2_batch, Y2_batch = X2_batch.to(device), Y2_batch.to(device)
            
            # Forward pass for both domains
            optimizer.zero_grad()
            
            # Domain 1
            outputs1 = model(X1_batch)
            loss1 = criterion(outputs1, Y1_batch)
            
            # Domain 2
            outputs2 = model(X2_batch)
            loss2 = criterion(outputs2, Y2_batch)
            
            # Average loss
            total_loss = (loss1 + loss2) / 2
            
            # Backward pass
            total_loss.backward()
            
            # Apply gradient alignment
            # Stack data from both domains for gradient alignment
            X_stacked = torch.cat([X1_batch, X2_batch], dim=0)
            Y_stacked = torch.cat([Y1_batch, Y2_batch], dim=0)
            
            # Forward pass with stacked data
            outputs_stacked = model(X_stacked)
            
            # Apply gradient alignment using the correct function signature
            # Reshape for gradient alignment: (n_envs, batch_size, features)
            batch_size_per_env = X1_batch.size(0)
            X_reshaped = torch.stack([X1_batch, X2_batch], dim=0)  # (2, batch_size, 3)
            Y_reshaped = torch.stack([Y1_batch, Y2_batch], dim=0)  # (2, batch_size, 1)
            
            # Forward pass with reshaped data
            outputs_reshaped = model(X_reshaped.view(-1, X_reshaped.size(-1)))  # Flatten for forward pass
            outputs_reshaped = outputs_reshaped.view(2, batch_size_per_env, 1)  # Reshape back
            
            mean_loss, masks = get_grads(
                agreement_threshold=DEFAULT_AGREEMENT_THRESHOLD,
                batch_size=batch_size_per_env,
                loss_fn=criterion,
                n_agreement_envs=2,  # 2 domains
                params=list(model.parameters()),
                output=outputs_reshaped,
                target=Y_reshaped,
                method='and_mask',
                scale_grad_inverse_sparsity=DEFAULT_SCALE_GRAD_INVERSE_SPARSITY
            )
            
            optimizer.step()
            
            # Calculate accuracy
            pred1 = (outputs1 > 0.5).float()
            pred2 = (outputs2 > 0.5).float()
            acc1 = (pred1 == Y1_batch).float().mean()
            acc2 = (pred2 == Y2_batch).float().mean()
            
            epoch_losses.append(total_loss.item())
            epoch_accuracies.append((acc1 + acc2).item() / 2)
            
            pbar.update(1)
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}', 'Acc': f'{epoch_accuracies[-1]:.4f}'})
        
        pbar.close()
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        
        # Store history
        overall_losses.append(avg_loss)
        overall_accuracies.append(avg_acc)
        
        # Calculate domain-specific metrics
        model.eval()
        with torch.no_grad():
            # Domain 1
            X1_tensor = torch.FloatTensor(X1_train).to(device)
            Y1_tensor = torch.FloatTensor(Y1_train_binary).unsqueeze(1).to(device)
            outputs1 = model(X1_tensor)
            loss1 = criterion(outputs1, Y1_tensor)
            pred1 = (outputs1 > 0.5).float()
            acc1 = (pred1 == Y1_tensor).float().mean()
            
            domain1_losses.append(loss1.item())
            domain1_accuracies.append(acc1.item())
            
            # Domain 2
            X2_tensor = torch.FloatTensor(X2_train).to(device)
            Y2_tensor = torch.FloatTensor(Y2_train_binary).unsqueeze(1).to(device)
            outputs2 = model(X2_tensor)
            loss2 = criterion(outputs2, Y2_tensor)
            pred2 = (outputs2 > 0.5).float()
            acc2 = (pred2 == Y2_tensor).float().mean()
            
            domain2_losses.append(loss2.item())
            domain2_accuracies.append(acc2.item())
        
        model.train()
        
        # Save model every 10 epochs and at the end
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            save_model(model, epoch + 1, "gradaligned")
        
        print(f'Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}')
    
    # Save training history
    training_history = {
        'overall_losses': overall_losses,
        'overall_accuracies': overall_accuracies,
        'domain1_losses': domain1_losses,
        'domain1_accuracies': domain1_accuracies,
        'domain2_losses': domain2_losses,
        'domain2_accuracies': domain2_accuracies
    }
    
    save_training_history(training_history, "gradaligned")
    
    return training_history

def save_training_history(history, method):
    """Save training history to files"""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Save loss history
    loss_file = os.path.join(PLOTS_DIR, f"loss_{method}.npz")
    np.savez(loss_file, **{k: v for k, v in history.items() if 'losses' in k})
    
    # Save accuracy history
    accuracy_file = os.path.join(PLOTS_DIR, f"accuracy_{method}.npz")
    np.savez(accuracy_file, **{k: v for k, v in history.items() if 'accuracies' in k})
    
    print(f"Training history saved for {method} method")

def load_training_history(method):
    """Load training history from files"""
    # Load loss history
    loss_file = os.path.join(PLOTS_DIR, f"loss_{method}.npz")
    accuracy_file = os.path.join(PLOTS_DIR, f"accuracy_{method}.npz")
    
    if not os.path.exists(loss_file) or not os.path.exists(accuracy_file):
        print(f"Training history files not found for {method} method")
        return None
    
    loss_data = np.load(loss_file)
    accuracy_data = np.load(accuracy_file)
    
    # Combine into single history dict
    history = {}
    for key in loss_data.keys():
        history[key] = loss_data[key]
    for key in accuracy_data.keys():
        history[key] = accuracy_data[key]
    
    return history

if __name__ == "__main__":
    # Test training functions
    print("Testing training functions...")
    
    # Create dummy data
    train_data = {
        'domain1': {
            'train': (np.random.randn(100, 3), np.random.choice([-1, 1], 100))
        },
        'domain2': {
            'train': (np.random.randn(100, 3), np.random.choice([-1, 1], 100))
        }
    }
    
    # Test ERM training (just a few epochs)
    print("\nTesting ERM training...")
    from models import DomainGeneralizationModel
    model = DomainGeneralizationModel()
    history = train_model_erm(model, train_data, epochs=2, batch_size=16)
    
    print("Training test complete!")
