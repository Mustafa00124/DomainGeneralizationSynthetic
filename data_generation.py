import os
import numpy as np
import scipy.optimize
from sklearn.model_selection import train_test_split
import torch
from constants import *

# ============================================================
# Global parameters (weights + targets)
# ============================================================

# Invariant targets
Z_INV_POS = +1.0
Z_INV_NEG = -1.0

# Specific targets
Z_SPU_POS = +2.0
Z_SPU_NEG = -2.0

# Invariant coefficients
ALPHA_POS, BETA_POS, GAMMA_POS = 0.8, 0.2, -0.3
ALPHA_NEG, BETA_NEG, GAMMA_NEG = 0.9, 0.1, -0.2

# Specific weights (a*x1 + b*x2 + c*x3)
G_POS_DOM1 = (1.0, 0.3, 0.4)
G_POS_DOM2 = (-0.95, -0.28, 0.1)
G_NEG_DOM1 = (0.9, 0.4, 0.8)
G_NEG_DOM2 = (-0.85, -0.37, 0.2)

# ============================================================
# Invariant functions
# ============================================================

def h_pos(x1, x2, x3):
    return x3**3 + ALPHA_POS*(x1**2 + x2**2) + BETA_POS*x1*x2 + GAMMA_POS

def h_neg(x1, x2, x3):
    return x3**3 + ALPHA_NEG*(x1**2 + x2**2) + BETA_NEG*x1*x2 + GAMMA_NEG

# ============================================================
# Specific functions
# ============================================================

def g_func(x1, x2, x3, coeffs):
    a, b, c = coeffs
    return a*x1 + b*x2 + c*x3

# ============================================================
# Sampling function
# ============================================================

def sample_point(y, domain, x1=None):
    """
    y ∈ {+1, -1}, domain ∈ {1, 2}
    Returns: (x1, x2, x3, z_inv, z_sp)
    """
    if x1 is None:
        x1 = np.random.uniform(-3, 3)

    # Pick invariant + specific
    if y == +1:
        h_func, z_inv_target = h_pos, Z_INV_POS
        coeffs = G_POS_DOM1 if domain == 1 else G_POS_DOM2
        z_sp_target = Z_SPU_POS
    else:
        h_func, z_inv_target = h_neg, Z_INV_NEG
        coeffs = G_NEG_DOM1 if domain == 1 else G_NEG_DOM2
        z_sp_target = Z_SPU_NEG

    a, b, c = coeffs

    # Express x2 in terms of x1, x3 from the specific constraint
    # a*x1 + b*x2 + c*x3 = z_sp_target
    def x2_from_x3(x3):
        return (z_sp_target - a*x1 - c*x3) / b

    # Residual for invariant equation
    def residual(x3):
        x2 = x2_from_x3(x3)
        return h_func(x1, x2, x3) - z_inv_target

    # Solve for x3
    try:
        sol = scipy.optimize.root_scalar(residual, bracket=[-10, 10], method='brentq')
        if not sol.converged:
            return None
        x3 = sol.root
        x2 = x2_from_x3(x3)
    except ValueError:
        return None

    # Compute z's
    z_inv = h_func(x1, x2, x3)
    z_sp = g_func(x1, x2, x3, coeffs)

    # Verify (margin 0.1)
    valid = (abs(z_inv - z_inv_target) < 0.1 and abs(z_sp - z_sp_target) < 0.1)
    if not valid:
        print(f"[INVALID] class={y}, domain={domain}, x1={x1:.2f}, "
              f"z_inv={z_inv:.2f}, z_sp={z_sp:.2f}")

    return x1, x2, x3, z_inv, z_sp

# ============================================================
# Generate dataset
# ============================================================

def generate_dataset(n_samples_per_class=DEFAULT_SAMPLES_PER_CLASS):
    """Generate synthetic dataset with 2 domains and 2 classes"""
    X, Y, D = [], [], []
    
    for domain in [1, 2]:
        for y in [-1, 1]:  # -1: negative class, 1: positive class
            for _ in range(n_samples_per_class):
                pt = sample_point(y, domain)
                if pt is not None:
                    X.append(pt[:3])  # only x1, x2, x3
                    Y.append(y)
                    D.append(domain)
    
    return np.array(X), np.array(Y), np.array(D)

# ============================================================
# Intervention datasets
# ============================================================

def make_intervention_datasets(X, Y, D):
    """
    Given original dataset (X, Y, D),
    build two counterfactual test sets:
      1. Invariant intervention (flip invariant target to other class)
      2. Spurious intervention (swap domain while keeping same invariant)
    Returns: (X_inv_int, Y_inv_int, D_inv_int), (X_spu_int, Y_spu_int, D_spu_int)
    """
    
    X_inv_int, X_spu_int = [], []
    Y_inv_int, D_inv_int = [], [] # Separate lists for invariant intervention
    Y_spu_int, D_spu_int = [], [] # Separate lists for spurious intervention

    for x, y, d in zip(X, Y, D):
        x1 = x[0]

        # --- Invariant intervention: flip class invariant but keep same spurious ---
        y_flip = -y  # flip class
        pt_inv = sample_point(y_flip, d, x1=x1)
        if pt_inv is not None:
            X_inv_int.append(pt_inv[:3])
            Y_inv_int.append(y)     # keep *original* label for evaluation
            D_inv_int.append(d)

        # --- Spurious intervention: swap domain while keeping same invariant ---
        d_swap = 2 if d == 1 else 1
        pt_spu = sample_point(y, d_swap, x1=x1)
        if pt_spu is not None:
            X_spu_int.append(pt_spu[:3])
            Y_spu_int.append(y)     # keep *original* label for evaluation
            D_spu_int.append(d_swap)

    return (np.array(X_inv_int), np.array(Y_inv_int), np.array(D_inv_int)), \
           (np.array(X_spu_int), np.array(Y_spu_int), np.array(D_spu_int))

# ============================================================
# Data preparation for training
# ============================================================

def prepare_data_for_training(X, Y, D, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE):
    """Split data by domain into train/test tensors"""
    # Split data by domain
    domain1_mask = (D == 1)
    domain2_mask = (D == 2)
    
    X1, Y1 = X[domain1_mask], Y[domain1_mask]
    X2, Y2 = X[domain2_mask], Y[domain2_mask]
    
    # Split each domain into train/test
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(
        X1, Y1, test_size=test_size, random_state=random_state, stratify=Y1
    )
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(
        X2, Y2, test_size=test_size, random_state=random_state, stratify=Y2
    )
    
    return {
        'domain1': {
            'train': (X1_train, Y1_train),
            'test': (X1_test, Y1_test)
        },
        'domain2': {
            'train': (X2_train, Y2_train),
            'test': (X2_test, Y2_test)
        }
    }

def save_generated_data(X, Y, D, intervention_data, train_test_data):
    """Save all generated data to files"""
    os.makedirs(GENERATED_DATA_DIR, exist_ok=True)
    
    # Save raw data
    np.savez(
        os.path.join(GENERATED_DATA_DIR, RAW_DATA_FILE),
        X=X, Y=Y, D=D
    )
    
    # Save intervention data
    (X_inv_int, Y_inv_int, D_inv_int), (X_spu_int, Y_spu_int, D_spu_int) = intervention_data
    np.savez(
        os.path.join(GENERATED_DATA_DIR, INTERVENTION_DATA_FILE),
        X_inv_int=X_inv_int, Y_inv_int=Y_inv_int, D_inv_int=D_inv_int,
        X_spu_int=X_spu_int, Y_spu_int=Y_spu_int, D_spu_int=D_spu_int
    )
    
    # Save train/test split data
    np.savez(
        os.path.join(GENERATED_DATA_DIR, TRAIN_TEST_SPLIT_FILE),
        X1_train=train_test_data['domain1']['train'][0],
        Y1_train=train_test_data['domain1']['train'][1],
        X1_test=train_test_data['domain1']['test'][0],
        Y1_test=train_test_data['domain1']['test'][1],
        X2_train=train_test_data['domain2']['train'][0],
        Y2_train=train_test_data['domain2']['train'][1],
        X2_test=train_test_data['domain2']['test'][0],
        Y2_test=train_test_data['domain2']['test'][1]
    )
    
    print(f"Data saved to {GENERATED_DATA_DIR}/")

def load_generated_data():
    """Load all generated data from files"""
    # Load raw data
    raw_data = np.load(os.path.join(GENERATED_DATA_DIR, RAW_DATA_FILE))
    X, Y, D = raw_data['X'], raw_data['Y'], raw_data['D']
    
    # Load intervention data
    intervention_data = np.load(os.path.join(GENERATED_DATA_DIR, INTERVENTION_DATA_FILE))
    inv_data = (intervention_data['X_inv_int'], intervention_data['Y_inv_int'], intervention_data['D_inv_int'])
    spu_data = (intervention_data['X_spu_int'], intervention_data['Y_spu_int'], intervention_data['D_spu_int'])
    
    # Load train/test split data
    split_data = np.load(os.path.join(GENERATED_DATA_DIR, TRAIN_TEST_SPLIT_FILE))
    train_test_data = {
        'domain1': {
            'train': (split_data['X1_train'], split_data['Y1_train']),
            'test': (split_data['X1_test'], split_data['Y1_test'])
        },
        'domain2': {
            'train': (split_data['X2_train'], split_data['Y2_train']),
            'test': (split_data['X2_test'], split_data['Y2_test'])
        }
    }
    
    return X, Y, D, (inv_data, spu_data), train_test_data

if __name__ == "__main__":
    # Generate data
    print("Generating synthetic dataset...")
    X, Y, D = generate_dataset(DEFAULT_SAMPLES_PER_CLASS)
    print(f"Generated {len(X)} samples")
    
    # Create intervention datasets
    print("Creating intervention datasets...")
    intervention_data = make_intervention_datasets(X, Y, D)
    (X_inv_int, Y_inv_int, D_inv_int), (X_spu_int, Y_spu_int, D_spu_int) = intervention_data
    print(f"Invariant intervention: {len(X_inv_int)} samples")
    print(f"Spurious intervention: {len(X_spu_int)} samples")
    
    # Prepare training data
    print("Preparing training data...")
    train_test_data = prepare_data_for_training(X, Y, D)
    
    # Save all data
    print("Saving data to files...")
    save_generated_data(X, Y, D, intervention_data, train_test_data)
    
    print("Data generation complete!")
