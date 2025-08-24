#!/usr/bin/env python3
"""
Main training script for Domain Generalization project.
This script orchestrates the entire workflow from data generation to model training and evaluation.
"""

import argparse
import os
import sys
import torch
from constants import *

# Import modules
from data_generation import generate_dataset, make_intervention_datasets, prepare_data_for_training, save_generated_data
from models import DomainGeneralizationModel
from training import train_model_erm, train_model_gradient_aligned
from plotting import (plot_inputs_by_domain, plot_inputs_3d_matplotlib, plot_inputs_3d_plotly,
                     plot_latents_by_domain, plot_training_history_detailed, plot_intervention_accuracy_table)
from testing import run_comprehensive_testing

def main():
    parser = argparse.ArgumentParser(description='Domain Generalization Training Pipeline')
    parser.add_argument('--method', type=str, default='both', 
                       choices=['erm', 'gradaligned', 'both'],
                       help='Training method to use')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--samples_per_class', type=int, default=DEFAULT_SAMPLES_PER_CLASS,
                       help='Number of samples per class per domain')
    parser.add_argument('--skip_data_generation', action='store_true',
                       help='Skip data generation if data already exists')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training if models already exist')
    parser.add_argument('--skip_plotting', action='store_true',
                       help='Skip plotting if plots already exist')
    parser.add_argument('--skip_testing', action='store_true',
                       help='Skip testing if already done')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DOMAIN GENERALIZATION TRAINING PIPELINE")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Samples per class: {args.samples_per_class}")
    print("=" * 60)
    
    # Step 1: Data Generation
    if not args.skip_data_generation:
        print("\nStep 1: Generating synthetic data...")
        try:
            # Generate dataset
            X, Y, D = generate_dataset(args.samples_per_class)
            print(f"Generated {len(X)} samples")
            
            # Create intervention datasets
            intervention_data = make_intervention_datasets(X, Y, D)
            (X_inv_int, Y_inv_int, D_inv_int), (X_spu_int, Y_spu_int, D_spu_int) = intervention_data
            print(f"Invariant intervention: {len(X_inv_int)} samples")
            print(f"Spurious intervention: {len(X_spu_int)} samples")
            
            # Prepare training data
            train_test_data = prepare_data_for_training(X, Y, D)
            
            # Save all data
            save_generated_data(X, Y, D, intervention_data, train_test_data)
            print("Data generation completed successfully!")
            
        except Exception as e:
            print(f"Data generation failed: {e}")
            return
    else:
        print("\nStep 1: Skipping data generation (data already exists)")
        try:
            from data_generation import load_generated_data
            X, Y, D, intervention_data, train_test_data = load_generated_data()
            print("Loaded existing data successfully!")
        except Exception as e:
            print(f"Failed to load existing data: {e}")
            return
    
    # Step 2: Input Space Visualization
    if not args.skip_plotting:
        print("\nStep 2: Creating input space visualizations...")
        try:
            plot_inputs_by_domain(X, Y, D)
            plot_inputs_3d_matplotlib(X, Y, D)
            plot_inputs_3d_plotly(X, Y, D)
            print("Input space visualizations completed!")
        except Exception as e:
            print(f"Input space visualization failed: {e}")
    
    # Step 3: Model Training
    if not args.skip_training:
        print("\nStep 3: Training models...")
        
        # Create models directory
        os.makedirs(os.path.join(MODELS_DIR, EPOCHS_DIR), exist_ok=True)
        
        if args.method in ['erm', 'both']:
            print("\nTraining ERM model...")
            try:
                model_erm = DomainGeneralizationModel()
                erm_history = train_model_erm(
                    model_erm, 
                    train_test_data, 
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr
                )
                print("ERM training completed successfully!")
            except Exception as e:
                print(f"ERM training failed: {e}")
                if args.method == 'erm':
                    return
        
        if args.method in ['gradaligned', 'both']:
            print("\nTraining Gradient Aligned model...")
            try:
                model_gradaligned = DomainGeneralizationModel()
                gradaligned_history = train_model_gradient_aligned(
                    model_gradaligned, 
                    train_test_data, 
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr
                )
                print("Gradient Aligned training completed successfully!")
            except Exception as e:
                print(f"Gradient Aligned training failed: {e}")
                if args.method == 'gradaligned':
                    return
    else:
        print("\nStep 3: Skipping training (models already exist)")
    
    # Step 4: Training History Visualization
    if not args.skip_plotting:
        print("\nStep 4: Creating training history visualizations...")
        try:
            if args.method in ['erm', 'both']:
                from training import load_training_history
                erm_history = load_training_history("erm")
                if erm_history:
                    try:
                        plot_training_history_detailed(erm_history, "erm")
                        print("ERM training history plots created!")
                    except Exception as e:
                        print(f"Warning: Could not create ERM plots: {e}")
                else:
                    print("Warning: Could not load ERM training history")
            
            if args.method in ['gradaligned', 'both']:
                from training import load_training_history
                gradaligned_history = load_training_history("gradaligned")
                if gradaligned_history:
                    try:
                        plot_training_history_detailed(gradaligned_history, "gradaligned")
                        print("Gradient Aligned training history plots created!")
                    except Exception as e:
                        print(f"Warning: Could not create Gradient Aligned plots: {e}")
                else:
                    print("Warning: Could not load Gradient Aligned training history")
            
            # Create intervention accuracy comparison if both methods were trained
            if args.method == 'both':
                try:
                    # Load both histories for comparison
                    erm_history = load_training_history("erm")
                    gradaligned_history = load_training_history("gradaligned")
                    
                    if erm_history and gradaligned_history:
                        # Get final accuracies for intervention comparison
                        erm_results = {
                            'invariant': {
                                'overall_acc': erm_history['overall_accuracies'][-1],
                                'domain1_acc': erm_history['domain1_accuracies'][-1],
                                'domain2_acc': erm_history['domain2_accuracies'][-1]
                            },
                            'spurious': {
                                'overall_acc': erm_history['overall_accuracies'][-1],
                                'domain1_acc': erm_history['domain1_accuracies'][-1],
                                'domain2_acc': erm_history['domain2_accuracies'][-1]
                            }
                        }
                        
                        gradaligned_results = {
                            'invariant': {
                                'overall_acc': gradaligned_history['overall_accuracies'][-1],
                                'domain1_acc': gradaligned_history['domain1_accuracies'][-1],
                                'domain2_acc': gradaligned_history['domain2_accuracies'][-1]
                            },
                            'spurious': {
                                'overall_acc': gradaligned_history['overall_accuracies'][-1],
                                'domain1_acc': gradaligned_history['domain1_accuracies'][-1],
                                'domain2_acc': gradaligned_history['domain2_accuracies'][-1]
                            }
                        }
                        
                        plot_intervention_accuracy_table(erm_results, gradaligned_results, "both")
                        print("Intervention accuracy comparison created!")
                except Exception as e:
                    print(f"Intervention accuracy comparison failed: {e}")
            
            print("Training history visualizations completed!")
        except Exception as e:
            print(f"Training history visualization failed: {e}")
    
    # Step 5: Latent Space Visualization (if models exist)
    if not args.skip_plotting:
        print("\nStep 5: Creating latent space visualizations...")
        try:
            # Load a trained model to get latent representations
            from models import list_saved_models, load_model
            
            model_files = list_saved_models()
            if model_files:
                # Use the first available model
                model_file = model_files[0]
                model, checkpoint = load_model(model_file)
                
                # Get latent representations
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    Z = model.get_latent_representation(X_tensor).numpy()
                
                plot_latents_by_domain(Z, Y, D)
                print("Latent space visualizations completed!")
            else:
                print("No trained models found for latent space visualization")
        except Exception as e:
            print(f"Latent space visualization failed: {e}")
    
    # Step 6: Model Testing
    if not args.skip_testing:
        print("\nStep 6: Testing trained models...")
        try:
            if args.method in ['erm', 'both']:
                print("\nTesting ERM model...")
                erm_results = run_comprehensive_testing("erm")
                if erm_results:
                    print("ERM testing completed successfully!")
            
            if args.method in ['gradaligned', 'both']:
                print("\nTesting Gradient Aligned model...")
                gradaligned_results = run_comprehensive_testing("gradaligned")
                if gradaligned_results:
                    print("Gradient Aligned testing completed successfully!")
            
            print("Model testing completed!")
        except Exception as e:
            print(f"Model testing failed: {e}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Print summary of generated files
    print("\nGenerated files and directories:")
    if os.path.exists(GENERATED_DATA_DIR):
        print(f"  - {GENERATED_DATA_DIR}/ (generated data)")
    
    if os.path.exists(MODELS_DIR):
        print(f"  - {MODELS_DIR}/ (trained models)")
    
    if os.path.exists(PLOTS_DIR):
        print(f"  - {PLOTS_DIR}/ (visualizations)")
        for subdir in ['input_space', 'latent_space', 'loss_accuracy']:
            subdir_path = os.path.join(PLOTS_DIR, subdir)
            if os.path.exists(subdir_path):
                files = os.listdir(subdir_path)
                print(f"    - {subdir}/ ({len(files)} files)")

if __name__ == "__main__":
    main()
