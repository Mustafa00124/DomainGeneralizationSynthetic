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
from plotting import (plot_inputs_by_domain, plot_training_history_detailed, 
                     plot_intervention_accuracy_table, plot_intervention_validation_history,
                     plot_intervention_accuracy_history, plot_intervention_inputs_by_domain)
from testing import run_comprehensive_testing

def main():
    parser = argparse.ArgumentParser(description='Domain Generalization Training Pipeline')
    parser.add_argument('--method', type=str, default='erm', 
                       choices=['erm', 'gradaligned'],
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
            X_inv_int, Y_inv_int, D_inv_int = intervention_data
            print(f"Invariant intervention: {len(X_inv_int)} samples")
            
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
            # Also create intervention input space visualization
            X_inv_int, Y_inv_int, D_inv_int = intervention_data
            plot_intervention_inputs_by_domain(X_inv_int, Y_inv_int, D_inv_int)
            print("Input space visualizations completed!")
        except Exception as e:
            print(f"Input space visualization failed: {e}")
    
    # Step 3: Model Training
    if not args.skip_training:
        print("\nStep 3: Training models...")
        
        # Create models directory
        os.makedirs(os.path.join(MODELS_DIR, EPOCHS_DIR), exist_ok=True)
        
        if args.method == 'erm':
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
        
        if args.method == 'gradaligned':
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
            if args.method == 'erm':
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
            
            if args.method == 'gradaligned':
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
            
            # Create intervention accuracy history plots for individual methods
            if args.method == 'erm':
                try:
                    erm_history = load_training_history("erm")
                    if erm_history:
                        plot_intervention_accuracy_history(erm_history, "erm")
                        print("ERM intervention accuracy history plot created!")
                except Exception as e:
                    print(f"ERM intervention accuracy history plot failed: {e}")
            
            if args.method == 'gradaligned':
                try:
                    gradaligned_history = load_training_history("gradaligned")
                    if gradaligned_history:
                        plot_intervention_accuracy_history(gradaligned_history, "gradaligned")
                        print("Gradient Aligned intervention accuracy history plot created!")
                except Exception as e:
                    print(f"Gradient Aligned intervention accuracy history plot failed: {e}")
            
            print("Training history visualizations completed!")
        except Exception as e:
            print(f"Training history visualization failed: {e}")
    
    # Step 5: Model Testing
    if not args.skip_testing:
        print("\nStep 5: Testing models...")
        try:
            run_comprehensive_testing(args.method, 'cpu')
            print("Model testing completed successfully!")
        except Exception as e:
            print(f"Model testing failed: {e}")
    else:
        print("\nStep 5: Skipping model testing (testing disabled)")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
