# Domain Generalization Project

A comprehensive implementation of domain generalization techniques for synthetic data, featuring both Empirical Risk Minimization (ERM) and Gradient Alignment optimization methods.

## Project Structure

The project is organized into modular components for better maintainability and flexibility:

```
DomainGeneralization/
├── constants.py              # Configuration constants and parameters
├── data_generation.py        # Synthetic data generation and intervention datasets
├── models.py                 # Neural network model definitions
├── training.py               # Training loops for ERM and gradient alignment
├── plotting.py               # Visualization functions for all plot types
├── testing.py                # Model evaluation and testing functions
├── main_training.py          # Main orchestration script
├── utils.py                  # Gradient alignment utility functions
├── generated_data/           # Saved datasets and splits
├── models/                   # Saved model checkpoints
│   └── epochs/              # Models saved at different epochs
├── plots/                    # Generated visualizations
│   ├── input_space/         # Input space plots (2D, 3D)
│   ├── latent_space/        # Latent space visualizations
│   ├── loss/                # Training loss plots
│   └── accuracy/            # Training accuracy and intervention plots
└── README.md                 # This file
```

## Key Features

- **Synthetic Data Generation**: Creates datasets with controlled invariant and spurious correlations
- **Intervention Datasets**: Generates counterfactual test sets for evaluating learned features
- **Dual Training Methods**: 
  - ERM (Empirical Risk Minimization)
  - Gradient Alignment optimization
- **Comprehensive Visualization**: 2D/3D plots, training curves, and intervention accuracy tables
- **Modular Architecture**: Each component can be run independently
- **Data Persistence**: All generated data, models, and plots are saved to disk
- **Flexible Execution**: Run individual components or the complete pipeline

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DomainGeneralization
```

2. Install required dependencies:
```bash
pip install torch numpy matplotlib plotly scipy scikit-learn tqdm
```

## Usage

### Running Individual Components

You can run each module independently for specific tasks:

#### 1. Data Generation
```bash
python data_generation.py
```
- Generates synthetic dataset with 2 domains and 2 classes
- Creates intervention datasets (invariant and spurious)
- Prepares train/test splits
- Saves all data to `generated_data/` folder

#### 2. Model Testing
```bash
python models.py
```
- Tests model creation and architecture
- Verifies model saving/loading functionality
- Tests forward pass and latent representation extraction

#### 3. Training Functions
```bash
python training.py
```
- Tests ERM training with dummy data
- Verifies training history saving/loading
- Tests gradient alignment optimization

#### 4. Visualization Functions
```bash
python plotting.py
```
- Tests all plotting functions with dummy data
- Creates sample visualizations
- Verifies plot saving functionality

#### 5. Testing Functions
```bash
python testing.py
```
- Tests model evaluation functions
- Tests intervention accuracy evaluation
- Tests model comparison and analysis functions

### Running the Complete Pipeline

Use the main training script to orchestrate the entire workflow:

```bash
# Train both ERM and Gradient Aligned models
python main_training.py --method both --epochs 500 --samples_per_class 50

# Train only ERM
python main_training.py --method erm --epochs 500

# Train only Gradient Aligned
python main_training.py --method gradaligned --epochs 500

# Skip certain steps if already completed
python main_training.py --method both --epochs 500 --skip_data_generation --skip_plotting
```

### Command Line Options

- `--method`: Training method (`erm`, `gradaligned`, or `both`)
- `--epochs`: Number of training epochs (default: 500)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--samples_per_class`: Samples per class per domain (default: 50)
- `--skip_data_generation`: Skip data generation if data exists
- `--skip_training`: Skip training if models exist
- `--skip_plotting`: Skip plotting if plots exist
- `--skip_testing`: Skip testing if already done

## Configuration

All configuration parameters are centralized in `constants.py`:

```python
# Training Parameters
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_AGREEMENT_THRESHOLD = 0.1
DEFAULT_SCALE_GRAD_INVERSE_SPARSITY = 1.0

# Data Generation Parameters
DEFAULT_SAMPLES_PER_CLASS = 50
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42

# Model Architecture
INPUT_SIZE = 3
HIDDEN1_SIZE = 50
HIDDEN2_SIZE = 2
OUTPUT_SIZE = 1
```

## Data Flow

1. **Data Generation** → `generated_data/` folder
2. **Training** → `models/epochs/` folder + training history files
3. **Plotting** → `plots/` folder with organized subdirectories
4. **Testing** → Reads from `generated_data/` and `models/` folders

## Output Files

### Generated Data
- `generated_data/raw_data.npz`: Original dataset (X, Y, D)
- `generated_data/intervention_data.npz`: Intervention datasets
- `generated_data/train_test_split.npz`: Train/test splits by domain

### Models
- `models/epochs/model_{method}_epoch_{epoch:04d}.pth`: Model checkpoints

### Plots
- **Input Space**: 2D grids, 3D matplotlib, interactive 3D plotly
- **Training History**: Loss and accuracy curves (overall and per-domain)
- **Intervention Accuracy**: Comparison tables between methods
- **Latent Space**: 2D visualizations of learned representations

## Key Concepts

### Domain Generalization
The project addresses the challenge of training models that generalize across different domains by learning invariant features while avoiding spurious correlations.

### Intervention Datasets
- **Invariant Intervention**: Tests if the model learned class-invariant features
- **Spurious Intervention**: Tests if the model learned domain-specific features

### Gradient Alignment
A training technique that only updates parameters in directions where both domains agree, preventing the learning of domain-specific features.

## Customization

### Adding New Training Methods
1. Implement training function in `training.py`
2. Add method to `main_training.py` argument parser
3. Update training orchestration logic

### Modifying Data Generation
1. Update functions in `data_generation.py`
2. Modify constants in `constants.py`
3. Update data saving/loading functions

### Adding New Visualizations
1. Implement plotting function in `plotting.py`
2. Add to main pipeline in `main_training.py`
3. Update plot directory creation

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Ensure all required packages are installed
2. **File Permissions**: Check write permissions for output directories
3. **Memory Issues**: Reduce batch size or samples per class for large datasets
4. **Training Convergence**: Adjust learning rate or agreement threshold

### Debug Mode
Run individual components with verbose output to identify issues:
```bash
python -v data_generation.py
python -v training.py
```

## Performance Notes

- **Data Generation**: Scales linearly with samples per class
- **Training**: ERM is faster than gradient alignment due to simpler optimization
- **Memory Usage**: Scales with batch size and model architecture
- **Plotting**: 3D plotly plots can be memory-intensive for large datasets

## Contributing

1. Follow the modular structure
2. Update constants in `constants.py` for new parameters
3. Add proper error handling and logging
4. Test individual components before integration
5. Update documentation for new features

## License

[Add your license information here]
