# Constants for Domain Generalization Project

# Training Parameters
DEFAULT_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_AGREEMENT_THRESHOLD = 1
DEFAULT_SCALE_GRAD_INVERSE_SPARSITY = 1.0

# Data Generation Parameters
DEFAULT_SAMPLES_PER_CLASS = 100
DEFAULT_VALIDATION_SAMPLES_PER_CLASS = 200
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42

# Model Architecture
INPUT_SIZE = 4
HIDDEN1_SIZE = 50
HIDDEN2_SIZE = 2
OUTPUT_SIZE = 1

# File Paths
GENERATED_DATA_DIR = "generated_data"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
EPOCHS_DIR = "epochs"

# Data Files
RAW_DATA_FILE = "raw_data.npz"
INTERVENTION_DATA_FILE = "intervention_data.npz"
TRAIN_TEST_SPLIT_FILE = "train_test_split.npz"

# Training History Files
LOSS_HISTORY_FILE = "loss_history.npz"
ACCURACY_HISTORY_FILE = "accuracy_history.npz"
