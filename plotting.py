import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import os
from constants import *

def create_plot_directories():
    """Create necessary directories for plots"""
    subdirs = ['input_space', 'latent_space', 'loss_accuracy']
    for subdir in subdirs:
        os.makedirs(os.path.join(PLOTS_DIR, subdir), exist_ok=True)

def plot_inputs_by_domain(X, Y, D):
    """Plot input space data with combined visualization"""
    create_plot_directories()
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Input Space Visualization - Combined Classes and Domains', fontsize=16)
    
    # Define projections
    projections = [(0, 1), (0, 2), (1, 2)]
    projection_names = ['x1 vs x2', 'x1 vs x3', 'x2 vs x3']
    
    for row, domain in enumerate([1, 2]):
        for col, (proj, name) in enumerate(zip(projections, projection_names)):
            ax = axes[row, col]
            
            # Plot positive class (class 1) with 'x' marker
            pos_mask = (Y == 1)
            domain_mask = (D == domain)
            pos_domain_mask = pos_mask & domain_mask
            
            if pos_domain_mask.sum() > 0:
                ax.scatter(X[pos_domain_mask, proj[0]], X[pos_domain_mask, proj[1]], 
                          c='blue', marker='x', s=50, alpha=0.7, label='Class 1')
            
            # Plot negative class (class -1) with 'o' marker
            neg_mask = (Y == -1)
            neg_domain_mask = neg_mask & domain_mask
            
            if neg_domain_mask.sum() > 0:
                ax.scatter(X[neg_domain_mask, proj[0]], X[neg_domain_mask, proj[1]], 
                          c='red', marker='o', s=50, alpha=0.7, label='Class -1')
            
            ax.set_xlabel(f'x{proj[0]+1}')
            ax.set_ylabel(f'x{proj[1]+1}')
            ax.set_title(f'Domain {domain}: {name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'input_space', 'input_space_combined_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Combined 2D input space plot saved!")

def plot_inputs_3d_matplotlib(X, Y, D):
    """Plot 3D input space with combined visualization"""
    create_plot_directories()
    
    fig = plt.figure(figsize=(15, 10))
    
    # Create 2x3 subplots for different projections
    for i, (proj, name) in enumerate([((0, 1, 2), 'x1 vs x2 vs x3')]):
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        # Plot positive class (class 1) with 'x' marker
        pos_mask = (Y == 1)
        pos_domain1_mask = pos_mask & (D == 1)
        pos_domain2_mask = pos_mask & (D == 2)
        
        if pos_domain1_mask.sum() > 0:
            ax.scatter(X[pos_domain1_mask, 0], X[pos_domain1_mask, 1], X[pos_domain1_mask, 2], 
                      c='blue', marker='x', s=50, alpha=0.7, label='Class 1, Domain 1')
        
        if pos_domain2_mask.sum() > 0:
            ax.scatter(X[pos_domain2_mask, 0], X[pos_domain2_mask, 1], X[pos_domain2_mask, 2], 
                      c='red', marker='x', s=50, alpha=0.7, label='Class 1, Domain 2')
        
        # Plot negative class (class -1) with 'o' marker
        neg_mask = (Y == -1)
        neg_domain1_mask = neg_mask & (D == 1)
        neg_domain2_mask = neg_mask & (D == 2)
        
        if neg_domain1_mask.sum() > 0:
            ax.scatter(X[neg_domain1_mask, 0], X[neg_domain1_mask, 1], X[neg_domain1_mask, 2], 
                      c='blue', marker='o', s=50, alpha=0.7, label='Class -1, Domain 1')
        
        if neg_domain2_mask.sum() > 0:
            ax.scatter(X[neg_domain2_mask, 0], X[neg_domain2_mask, 1], X[neg_domain2_mask, 2], 
                      c='red', marker='o', s=50, alpha=0.7, label='Class -1, Domain 2')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.set_title(f'3D Input Space: {name}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'input_space', 'input_space_3d_matplotlib.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Combined 3D matplotlib input space plot saved!")

def plot_inputs_3d_plotly(X, Y, D):
    """Plot 3D input space with plotly for interactivity"""
    create_plot_directories()
    
    fig = go.Figure()
    
    # Plot positive class (class 1) with 'x' marker
    pos_mask = (Y == 1)
    pos_domain1_mask = pos_mask & (D == 1)
    pos_domain2_mask = pos_mask & (D == 2)
    
    if pos_domain1_mask.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=X[pos_domain1_mask, 0], y=X[pos_domain1_mask, 1], z=X[pos_domain1_mask, 2],
            mode='markers', marker=dict(symbol='x', size=5, color='blue', opacity=0.7),
            name='Class 1, Domain 1'
        ))
    
    if pos_domain2_mask.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=X[pos_domain2_mask, 0], y=X[pos_domain2_mask, 1], z=X[pos_domain2_mask, 2],
            mode='markers', marker=dict(symbol='x', size=5, color='red', opacity=0.7),
            name='Class 1, Domain 2'
        ))
    
    # Plot negative class (class -1) with 'o' marker
    neg_mask = (Y == -1)
    neg_domain1_mask = neg_mask & (D == 1)
    neg_domain2_mask = neg_mask & (D == 2)
    
    if neg_domain1_mask.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=X[neg_domain1_mask, 0], y=X[neg_domain1_mask, 1], z=X[neg_domain1_mask, 2],
            mode='markers', marker=dict(symbol='circle', size=5, color='blue', opacity=0.7),
            name='Class -1, Domain 1'
        ))
    
    if neg_domain2_mask.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=X[neg_domain2_mask, 0], y=X[neg_domain2_mask, 1], z=X[neg_domain2_mask, 2],
            mode='markers', marker=dict(symbol='circle', size=5, color='red', opacity=0.7),
            name='Class -1, Domain 2'
        ))
    
    fig.update_layout(
        title='3D Input Space Visualization - Combined Classes and Domains',
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='x3'
        ),
        width=800, height=600
    )
    
    fig.write_html(os.path.join(PLOTS_DIR, 'input_space', 'input_space_3d_plotly.html'))
    print("Combined 3D plotly input space plot saved!")

def plot_latents_by_domain(Z, Y, D):
    """Plot latent space with combined visualization"""
    create_plot_directories()
    
    # Create 2x1 subplots for both domains
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Latent Space Visualization - Combined Classes and Domains', fontsize=16)
    
    for row, domain in enumerate([1, 2]):
        ax = axes[row]
        
        # Plot positive class (class 1) with 'x' marker
        pos_mask = (Y == 1)
        domain_mask = (D == domain)
        pos_domain_mask = pos_mask & domain_mask
        
        if pos_domain_mask.sum() > 0:
            ax.scatter(Z[pos_domain_mask, 0], Z[pos_domain_mask, 1], 
                      c='blue', marker='x', s=50, alpha=0.7, label='Class 1')
        
        # Plot negative class (class -1) with 'o' marker
        neg_mask = (Y == -1)
        neg_domain_mask = neg_mask & domain_mask
        
        if neg_domain_mask.sum() > 0:
            ax.scatter(Z[neg_domain_mask, 0], Z[neg_domain_mask, 1], 
                      c='red', marker='o', s=50, alpha=0.7, label='Class -1')
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_title(f'Domain {domain}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'latent_space', 'latent_space_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Combined latent space plot saved!")

def plot_training_history_detailed(history, method):
    """Plot training history with combined loss and accuracy plots"""
    create_plot_directories()
    
    epochs = range(1, len(history['overall_losses']) + 1)
    
    # Create combined loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['overall_losses'], 'b-', linewidth=2, label='Overall Loss')
    plt.plot(epochs, history['domain1_losses'], 'g--', linewidth=2, label='Domain 1 Loss')
    plt.plot(epochs, history['domain2_losses'], 'r--', linewidth=2, label='Domain 2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{method.upper()} Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create combined accuracy plot
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
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_accuracy', f'loss_{method}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_accuracy', f'accuracy_{method}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined training history plots saved for {method} method!")

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
    
    plt.title('Intervention Accuracy Comparison', fontsize=16, pad=20)
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
    
    # Create dummy data for testing
    np.random.seed(42)
    X = np.random.randn(200, 3)
    Y = np.random.choice([-1, 1], 200)
    D = np.random.choice([1, 2], 200)
    
    # Test input space plotting
    print("\nTesting input space plotting...")
    plot_inputs_by_domain(X, Y, D)
    plot_inputs_3d_matplotlib(X, Y, D)
    plot_inputs_3d_plotly(X, Y, D)
    
    # Test latent space plotting
    print("\nTesting latent space plotting...")
    Z = np.random.randn(200, 2)  # Dummy latent representations
    plot_latents_by_domain(Z, Y, D)
    
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
