import matplotlib.pyplot as plt
import numpy as np
import os
from constants import *

def create_plot_directories():
    """Create necessary directories for plots"""
    subdirs = ['input_space', 'loss_accuracy']
    for subdir in subdirs:
        os.makedirs(os.path.join(PLOTS_DIR, subdir), exist_ok=True)

def plot_inputs_by_domain(X, Y, D):
    """Plot input space data: x1 vs x2 and x3 vs x4 for each domain"""
    create_plot_directories()
    
    # Create figure with 2x2 subplots: (x1 vs x2) and (x3 vs x4) for each domain
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Input Space Visualization - Combined Classes and Domains', fontsize=16)
    
    for row, domain in enumerate([1, 2]):
        # Plot 1: x1 vs x2 (2D scatter)
        ax1 = axes[row, 0]
        
        # Plot positive class (class 1) with 'x' marker
        pos_mask = (Y == 1)
        domain_mask = (D == domain)
        pos_domain_mask = pos_mask & domain_mask
        
        if pos_domain_mask.sum() > 0:
            ax1.scatter(X[pos_domain_mask, 0], X[pos_domain_mask, 1], 
                      c='blue', marker='x', s=50, alpha=0.7, label='Class 1')
        
        # Plot negative class (class -1) with 'o' marker
        neg_mask = (Y == -1)
        neg_domain_mask = neg_mask & domain_mask
        
        if neg_domain_mask.sum() > 0:
            ax1.scatter(X[neg_domain_mask, 0], X[neg_domain_mask, 1], 
                      c='red', marker='o', s=50, alpha=0.7, label='Class -1')
        
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_title(f'Domain {domain}: x1 vs x2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: x3 vs x4 (2D scatter)
        ax2 = axes[row, 1]
        
        # Plot positive class x3 vs x4
        if pos_domain_mask.sum() > 0:
            ax2.scatter(X[pos_domain_mask, 2], X[pos_domain_mask, 3], 
                      c='blue', marker='x', s=50, alpha=0.7, label='Class 1')
        
        # Plot negative class x3 vs x4
        if neg_domain_mask.sum() > 0:
            ax2.scatter(X[neg_domain_mask, 2], X[neg_domain_mask, 3], 
                      c='red', marker='o', s=50, alpha=0.7, label='Class -1')
        
        ax2.set_xlabel('x3')
        ax2.set_ylabel('x4')
        ax2.set_title(f'Domain {domain}: x3 vs x4')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'input_space', 'input_space_combined_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Combined 2D input space plot saved!")

def plot_intervention_inputs_by_domain(X_inv_int, Y_inv_int, D_inv_int):
    """Plot intervention input space data: x1 vs x2 and x3 vs x4 for each domain"""
    create_plot_directories()
    
    # Create figure with 2x2 subplots: (x1 vs x2) and (x3 vs x4) for each domain
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Intervention Input Space Visualization - Combined Classes and Domains', fontsize=16)
    
    for row, domain in enumerate([1, 2]):
        # Plot 1: x1 vs x2 (2D scatter)
        ax1 = axes[row, 0]
        
        # Plot positive class (class 1) with 'x' marker
        pos_mask = (Y_inv_int == 1)
        domain_mask = (D_inv_int == domain)
        pos_domain_mask = pos_mask & domain_mask
        
        if pos_domain_mask.sum() > 0:
            ax1.scatter(X_inv_int[pos_domain_mask, 0], X_inv_int[pos_domain_mask, 1], 
                      c='blue', marker='x', s=50, alpha=0.7, label='Class 1')
        
        # Plot negative class (class -1) with 'o' marker
        neg_mask = (Y_inv_int == -1)
        neg_domain_mask = neg_mask & domain_mask
        
        if neg_domain_mask.sum() > 0:
            ax1.scatter(X_inv_int[neg_domain_mask, 0], X_inv_int[neg_domain_mask, 1], 
                      c='red', marker='o', s=50, alpha=0.7, label='Class -1')
        
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_title(f'Domain {domain}: x1 vs x2 (Intervention)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: x3 vs x4 (2D scatter)
        ax2 = axes[row, 1]
        
        # Plot positive class x3 vs x4
        if pos_domain_mask.sum() > 0:
            ax2.scatter(X_inv_int[pos_domain_mask, 2], X_inv_int[pos_domain_mask, 3], 
                      c='blue', marker='x', s=50, alpha=0.7, label='Class 1')
        
        # Plot negative class x3 vs x4
        if neg_domain_mask.sum() > 0:
            ax2.scatter(X_inv_int[neg_domain_mask, 2], X_inv_int[neg_domain_mask, 3], 
                      c='red', marker='o', s=50, alpha=0.7, label='Class -1')
        
        ax2.set_xlabel('x3')
        ax2.set_ylabel('x4')
        ax2.set_title(f'Domain {domain}: x3 vs x4 (Intervention)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'input_space', 'intervention_input_space_combined_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Intervention input space plot saved!")

def plot_training_history_detailed(history, method):
    """Plot training history with combined loss and accuracy plots in a single PNG"""
    create_plot_directories()
    
    epochs = range(1, len(history['overall_losses']) + 1)
    
    # Create single figure with both loss and accuracy plots side by side
    plt.figure(figsize=(16, 6))
    
    # Loss plot on the left
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['overall_losses'], 'b-', linewidth=2, label='Overall Loss')
    plt.plot(epochs, history['domain1_losses'], 'g--', linewidth=2, label='Domain 1 Loss')
    plt.plot(epochs, history['domain2_losses'], 'r--', linewidth=2, label='Domain 2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{method.upper()} Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot on the right
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['overall_accuracies'], 'b-', linewidth=2, label='Overall Accuracy')
    plt.plot(epochs, history['domain1_accuracies'], 'g--', linewidth=2, label='Domain 1 Accuracy')
    plt.plot(epochs, history['domain2_accuracies'], 'r--', linewidth=2, label='Domain 2 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{method.upper()} Training Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save with new naming convention
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_accuracy', f'{method}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined training history plot saved for {method} method!")

def plot_intervention_accuracy_history(history, method):
    """Plot intervention accuracy per epoch for a single method"""
    create_plot_directories()
    
    if 'invariant_accuracies' not in history:
        print(f"No intervention data available for {method}")
        return
    
    epochs = range(1, len(history['invariant_accuracies']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['invariant_accuracies'], 'b-', linewidth=2, label=f'{method.upper()} Intervention Accuracy')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Chance (50%)')
    plt.xlabel('Epoch')
    plt.ylabel('Intervention Accuracy')
    plt.title(f'{method.upper()} Intervention Accuracy Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_accuracy', f'{method}_intervention_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Intervention accuracy plot saved for {method} method!")

def plot_intervention_validation_history(erm_history, gradaligned_history, method):
    """Plot intervention validation history comparing both methods"""
    create_plot_directories()
    
    # Create single figure with intervention accuracy comparison
    plt.figure(figsize=(12, 6))
    
    if 'invariant_accuracies' in erm_history and 'invariant_accuracies' in gradaligned_history:
        epochs = range(1, len(erm_history['invariant_accuracies']) + 1)
        plt.plot(epochs, erm_history['invariant_accuracies'], 'b-', linewidth=2, label='ERM Intervention')
        plt.plot(epochs, gradaligned_history['invariant_accuracies'], 'r-', linewidth=2, label='GradAligned Intervention')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Chance (50%)')
        plt.xlabel('Epoch')
        plt.ylabel('Intervention Accuracy')
        plt.title(f'{method.upper()} Intervention Accuracy Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
    else:
        plt.text(0.5, 0.5, 'No intervention validation data available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{method.upper()} Intervention Accuracy Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_accuracy', f'intervention_comparison_{method}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Intervention comparison plot saved for {method} method!")

def plot_intervention_accuracy_table(erm_results, gradaligned_results, method):
    """Plot intervention accuracy comparison as a table"""
    create_plot_directories()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    data = [
        ['Method', 'Overall', 'Domain 1', 'Domain 2'],
        ['ERM - Invariant', f"{erm_results['invariant']['overall_acc']:.4f}", 
         f"{erm_results['invariant']['domain1_acc']:.4f}", 
         f"{erm_results['invariant']['domain2_acc']:.4f}"],
        ['ERM - Spurious', f"{erm_results['spurious']['overall_acc']:.4f}", 
         f"{erm_results['spurious']['domain1_acc']:.4f}", 
         f"{erm_results['spurious']['domain2_acc']:.4f}"],
        ['GradAligned - Invariant', f"{gradaligned_results['invariant']['overall_acc']:.4f}", 
         f"{gradaligned_results['invariant']['domain1_acc']:.4f}", 
         f"{gradaligned_results['invariant']['domain2_acc']:.4f}"],
        ['GradAligned - Spurious', f"{gradaligned_results['spurious']['overall_acc']:.4f}", 
         f"{gradaligned_results['spurious']['domain1_acc']:.4f}", 
         f"{gradaligned_results['spurious']['domain2_acc']:.4f}"]
    ]
    
    # Create table
    table = ax.table(cellText=data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(data)):
        for j in range(len(data[0])):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#E8F5E8')
    
    plt.title(f'{method.upper()} Intervention Accuracy Comparison', fontsize=16, pad=20)
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_accuracy', f'intervention_accuracy_{method}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Intervention accuracy table saved for {method} method!")

def load_training_history_for_plotting(method):
    """Load training history for plotting"""
    from training import load_training_history
    return load_training_history(method)

if __name__ == "__main__":
    # Test plotting functions
    print("Testing plotting functions...")
    
    # Create dummy data for testing (4 features)
    np.random.seed(42)
    X = np.random.randn(200, 4)
    Y = np.random.choice([-1, 1], 200)
    D = np.random.choice([1, 2], 200)
    
    # Test input space plotting
    print("\nTesting input space plotting...")
    plot_inputs_by_domain(X, Y, D)
    
    # Test training history plotting
    print("\nTesting training history plotting...")
    dummy_history = {
        'overall_losses': [0.8, 0.6, 0.4, 0.3, 0.2],
        'overall_accuracies': [0.5, 0.7, 0.8, 0.85, 0.9],
        'domain1_losses': [0.9, 0.7, 0.5, 0.4, 0.3],
        'domain1_accuracies': [0.4, 0.6, 0.7, 0.8, 0.85],
        'domain2_losses': [0.7, 0.5, 0.3, 0.2, 0.1],
        'domain2_accuracies': [0.6, 0.8, 0.9, 0.9, 0.95]
    }
    plot_training_history_detailed(dummy_history, "test")
    
    # Test intervention accuracy table
    print("\nTesting intervention accuracy table...")
    dummy_erm = {
        'invariant': {'overall_acc': 0.85, 'domain1_acc': 0.8, 'domain2_acc': 0.9},
        'spurious': {'overall_acc': 0.75, 'domain1_acc': 0.7, 'domain2_acc': 0.8}
    }
    dummy_gradaligned = {
        'invariant': {'overall_acc': 0.9, 'domain1_acc': 0.85, 'domain2_acc': 0.95},
        'spurious': {'overall_acc': 0.8, 'domain1_acc': 0.75, 'domain2_acc': 0.85}
    }
    plot_intervention_accuracy_table(dummy_erm, dummy_gradaligned, "test")
    
    print("Plotting test complete!")
