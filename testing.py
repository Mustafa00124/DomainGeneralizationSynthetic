import torch
import torch.nn as nn
import numpy as np
import os
from constants import *
from models import load_model, list_saved_models
from data_generation import load_generated_data

def evaluate_model_on_test_data(model, test_data, device='cpu'):
    """Evaluate model performance on test data"""
    model.eval()
    criterion = nn.BCELoss()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    domain1_correct = 0
    domain1_total = 0
    domain2_correct = 0
    domain2_total = 0
    
    with torch.no_grad():
        # Domain 1
        X1_test, Y1_test = test_data['domain1']['test']
        X1_tensor = torch.FloatTensor(X1_test).to(device)
        Y1_tensor = torch.FloatTensor((Y1_test + 1) / 2).unsqueeze(1).to(device)
        
        outputs1 = model(X1_tensor)
        loss1 = criterion(outputs1, Y1_tensor)
        pred1 = (outputs1 > 0.5).float()
        correct1 = (pred1 == Y1_tensor).sum().item()
        
        total_loss += loss1.item()
        total_correct += correct1
        total_samples += Y1_tensor.size(0)
        domain1_correct += correct1
        domain1_total += Y1_tensor.size(0)
        
        # Domain 2
        X2_test, Y2_test = test_data['domain2']['test']
        X2_tensor = torch.FloatTensor(X2_test).to(device)
        Y2_tensor = torch.FloatTensor((Y2_test + 1) / 2).unsqueeze(1).to(device)
        
        outputs2 = model(X2_tensor)
        loss2 = criterion(outputs2, Y2_tensor)
        pred2 = (outputs2 > 0.5).float()
        correct2 = (pred2 == Y2_tensor).sum().item()
        
        total_loss += loss2.item()
        total_correct += correct2
        total_samples += Y2_tensor.size(0)
        domain2_correct += correct2
        domain2_total += Y2_tensor.size(0)
    
    # Calculate metrics
    avg_loss = total_loss / 2
    overall_accuracy = total_correct / total_samples
    domain1_accuracy = domain1_correct / domain1_total
    domain2_accuracy = domain2_correct / domain2_total
    
    return {
        'overall_loss': avg_loss,
        'overall_accuracy': overall_accuracy,
        'domain1_accuracy': domain1_accuracy,
        'domain2_accuracy': domain2_accuracy
    }

def evaluate_intervention_accuracy(model, intervention_data, device='cpu'):
    """Evaluate model on intervention datasets"""
    model.eval()
    criterion = nn.BCELoss()
    
    X_inv_int, Y_inv_int, D_inv_int = intervention_data
    
    # Convert labels from [-1, 1] to [0, 1]
    Y_inv_int_binary = (Y_inv_int + 1) / 2
    
    results = {}
    
    # Evaluate on invariant intervention
    if len(X_inv_int) > 0:
        X_inv_tensor = torch.FloatTensor(X_inv_int).to(device)
        Y_inv_tensor = torch.FloatTensor(Y_inv_int_binary).unsqueeze(1).to(device)
        
        outputs_inv = model(X_inv_tensor)
        loss_inv = criterion(outputs_inv, Y_inv_tensor)
        pred_inv = (outputs_inv > 0.5).float()
        correct_inv = (pred_inv == Y_inv_tensor).sum().item()
        
        # Calculate domain-specific accuracies
        domain1_mask = (D_inv_int == 1)
        domain2_mask = (D_inv_int == 2)
        
        domain1_correct = ((pred_inv == Y_inv_tensor) & (torch.BoolTensor(domain1_mask).unsqueeze(1).to(device))).sum().item()
        domain2_correct = ((pred_inv == Y_inv_tensor) & (torch.BoolTensor(domain2_mask).unsqueeze(1).to(device))).sum().item()
        
        domain1_total = domain1_mask.sum()
        domain2_total = domain2_mask.sum()
        
        results['invariant'] = {
            'overall_acc': correct_inv / len(X_inv_int),
            'domain1_acc': domain1_correct / domain1_total if domain1_total > 0 else 0,
            'domain2_acc': domain2_correct / domain2_total if domain2_total > 0 else 0
        }
    else:
        results['invariant'] = {'overall_acc': 0, 'domain1_acc': 0, 'domain2_acc': 0}
    
    # For backward compatibility, use the same results for spurious intervention
    results['spurious'] = results['invariant'].copy()
    
    return results

def compare_models_at_epochs(method, epochs_to_test, device='cpu'):
    """Compare model performance at different epochs"""
    print(f"Comparing {method} models at different epochs...")
    
    # Load test data
    X, Y, D, intervention_data, train_test_data = load_generated_data()
    
    results = {}
    
    for epoch in epochs_to_test:
        # Find model file for this epoch
        model_files = list_saved_models()
        target_file = None
        
        for model_file in model_files:
            if f"model_{method}_epoch_{epoch:04d}.pth" in model_file:
                target_file = model_file
                break
        
        if target_file is None:
            print(f"Model for epoch {epoch} not found, skipping...")
            continue
        
        # Load and evaluate model
        model, checkpoint = load_model(target_file, device)
        
        # Test on regular test data
        test_results = evaluate_model_on_test_data(model, train_test_data, device)
        
        # Test on intervention data
        intervention_results = evaluate_intervention_accuracy(model, intervention_data, device)
        
        results[epoch] = {
            'test': test_results,
            'intervention': intervention_results
        }
        
        print(f"Epoch {epoch}: Test Acc={test_results['overall_accuracy']:.4f}, "
              f"Inv Acc={intervention_results['invariant']['overall_acc']:.4f}, "
              f"Spu Acc={intervention_results['spurious']['overall_acc']:.4f}")
    
    return results

def analyze_training_progression(method, device='cpu'):
    """Analyze how model performance changes during training"""
    print(f"Analyzing training progression for {method}...")
    
    # Load test data
    X, Y, D, intervention_data, train_test_data = load_generated_data()
    
    # Get all saved models for this method
    model_files = list_saved_models()
    method_models = [f for f in model_files if f"model_{method}_epoch_" in f]
    
    if not method_models:
        print(f"No saved models found for {method} method")
        return None
    
    # Sort by epoch
    method_models.sort()
    
    progression = {}
    
    for model_file in method_models:
        # Extract epoch from filename
        epoch_str = model_file.split('epoch_')[1].split('.')[0]
        epoch = int(epoch_str)
        
        # Load and evaluate model
        model, checkpoint = load_model(model_file, device)
        
        # Test on regular test data
        test_results = evaluate_model_on_test_data(model, train_test_data, device)
        
        # Test on intervention data
        intervention_results = evaluate_intervention_accuracy(model, intervention_data, device)
        
        progression[epoch] = {
            'test': test_results,
            'intervention': intervention_results
        }
        
        print(f"Epoch {epoch}: Test Acc={test_results['overall_accuracy']:.4f}, "
              f"Inv Acc={intervention_results['invariant']['overall_acc']:.4f}, "
              f"Spu Acc={intervention_results['spurious']['overall_acc']:.4f}")
    
    return progression

def run_comprehensive_testing(method, device='cpu'):
    """Run comprehensive testing on a trained model"""
    print(f"Running comprehensive testing for {method} method...")
    
    # Load test data
    X, Y, D, intervention_data, train_test_data = load_generated_data()
    
    # Find the latest model for this method
    model_files = list_saved_models()
    method_models = [f for f in model_files if f"model_{method}_epoch_" in f]
    
    if not method_models:
        print(f"No saved models found for {method} method")
        return None
    
    # Get the latest model (highest epoch)
    latest_model = max(method_models, key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
    epoch = int(latest_model.split('epoch_')[1].split('.')[0])
    
    print(f"Testing latest {method} model (epoch {epoch})...")
    
    # Load and evaluate model
    model, checkpoint = load_model(latest_model, device)
    
    # Test on regular test data
    test_results = evaluate_model_on_test_data(model, train_test_data, device)
    
    # Test on intervention data
    intervention_results = evaluate_intervention_accuracy(model, intervention_data, device)
    
    # Print results
    print(f"\n========== {method.upper()} Model Results (Epoch {epoch}) ==========")
    print(f"Test Accuracy - Overall: {test_results['overall_accuracy']:.4f}")
    print(f"Test Accuracy - Domain 1: {test_results['domain1_accuracy']:.4f}")
    print(f"Test Accuracy - Domain 2: {test_results['domain2_accuracy']:.4f}")
    print(f"Invariant Intervention - Overall: {intervention_results['invariant']['overall_acc']:.4f}")
    print(f"Invariant Intervention - Domain 1: {intervention_results['invariant']['domain1_acc']:.4f}")
    print(f"Invariant Intervention - Domain 2: {intervention_results['invariant']['domain2_acc']:.4f}")

    
    return {
        'epoch': epoch,
        'test': test_results,
        'intervention': intervention_results
    }

if __name__ == "__main__":
    # Test testing functions
    print("Testing testing functions...")
    
    # Test model listing
    print("\nTesting model listing...")
    models = list_saved_models()
    print(f"Found {len(models)} saved models")
    
    # Test comprehensive testing if models exist
    if models:
        print("\nTesting comprehensive testing...")
        # Try to test with the first available method
        first_model = models[0]
        method = first_model.split('model_')[1].split('_epoch_')[0]
        print(f"Testing with {method} method...")
        
        try:
            results = run_comprehensive_testing(method)
            if results:
                print("Comprehensive testing completed successfully!")
        except Exception as e:
            print(f"Comprehensive testing failed: {e}")
    else:
        print("No models found for testing")
    
    print("Testing test complete!")
