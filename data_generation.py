import os
import numpy as np
from sklearn.model_selection import train_test_split
from constants import *

# ============================================================
# Global parameters
# ============================================================

Z_SPU_POS = +50.0
Z_SPU_NEG = -50.0
Z_INV_POS = +0.5
Z_INV_NEG = -0.5

# Specific coefficients (domain-dependent)
SPECIFIC_COEFFS = {
    1: (1.0, 0.5),
    2: (-0.8, 0.7)
}

# Invariant coefficients (shared)
INV_ALPHA = 0.05   # multiplies x1
INV_BETA  = 0.1    # multiplies x2
INV_GAMMA = 0.2    # multiplies x3
INV_DELTA = -0.15  # multiplies x4

RANGE_LINEAR = (1, 50)

# ============================================================
# Functions
# ============================================================

def g_func(x1, x2, coeffs):
    a, b = coeffs
    return a * x1 + b * x2

def h_func(x1, x2, x3, x4):
    return INV_ALPHA*x1 + INV_BETA*x2 + INV_GAMMA*x3 + INV_DELTA*x4

# ============================================================
# Sampling
# ============================================================

def sample_point(domain,z_sp,z_inv):
    """Sample (x1,x2,x3,x4) satisfying both constraints"""
    a, b = SPECIFIC_COEFFS[domain]

    for attempt in range(1000):
        branch = np.random.choice(range(8))

        if branch == 0:
            # Sample x1,x3 -> solve x2 from g, x4 from h
            x1 = np.random.uniform(*RANGE_LINEAR)
            x3 = np.random.uniform(*RANGE_LINEAR)
            x2 = (z_sp - a*x1)/b
            denom = INV_DELTA
            if abs(denom) < 1e-8: continue
            x4 = (z_inv - INV_ALPHA*x1 - INV_BETA*x2 - INV_GAMMA*x3)/denom

        elif branch == 1:
            # Sample x1,x4 -> solve x2, then x3
            x1 = np.random.uniform(*RANGE_LINEAR)
            x4 = np.random.uniform(*RANGE_LINEAR)
            x2 = (z_sp - a*x1)/b
            denom = INV_GAMMA
            if abs(denom) < 1e-8: continue
            x3 = (z_inv - INV_ALPHA*x1 - INV_BETA*x2 - INV_DELTA*x4)/denom

        elif branch == 2:
            # Sample x2,x3 -> solve x1, then x4
            x2 = np.random.uniform(*RANGE_LINEAR)
            x3 = np.random.uniform(*RANGE_LINEAR)
            x1 = (z_sp - b*x2)/a
            denom = INV_DELTA
            if abs(denom) < 1e-8: continue
            x4 = (z_inv - INV_ALPHA*x1 - INV_BETA*x2 - INV_GAMMA*x3)/denom

        elif branch == 3:
            # Sample x2,x4 -> solve x1, then x3
            x2 = np.random.uniform(*RANGE_LINEAR)
            x4 = np.random.uniform(*RANGE_LINEAR)
            x1 = (z_sp - b*x2)/a
            denom = INV_GAMMA
            if abs(denom) < 1e-8: continue
            x3 = (z_inv - INV_ALPHA*x1 - INV_BETA*x2 - INV_DELTA*x4)/denom

        elif branch == 4:
            # Sample x3,x4 -> solve x1, then x2
            x3 = np.random.uniform(*RANGE_LINEAR)
            x4 = np.random.uniform(*RANGE_LINEAR)
            denom = a*INV_BETA + b*INV_ALPHA
            if abs(denom) < 1e-8: continue
            x1 = (z_sp*INV_BETA + b*z_inv - b*INV_GAMMA*x3 - b*INV_DELTA*x4)/(denom)
            x2 = (z_sp - a*x1)/b

        elif branch == 5:
            # Sample x1,x2 -> solve x3, then x4
            x1 = np.random.uniform(*RANGE_LINEAR)
            x2 = (z_sp - a*x1)/b
            denom = INV_DELTA
            if abs(denom) < 1e-8: continue
            # pick x3 random, solve x4
            x3 = np.random.uniform(*RANGE_LINEAR)
            x4 = (z_inv - INV_ALPHA*x1 - INV_BETA*x2 - INV_GAMMA*x3)/denom

        elif branch == 6:
            # Sample x1,x2 -> solve x4, then x3
            x1 = np.random.uniform(*RANGE_LINEAR)
            x2 = (z_sp - a*x1)/b
            denom = INV_GAMMA
            if abs(denom) < 1e-8: continue
            # pick x4 random, solve x3
            x4 = np.random.uniform(*RANGE_LINEAR)
            x3 = (z_inv - INV_ALPHA*x1 - INV_BETA*x2 - INV_DELTA*x4)/denom

        else:
            # Sample x3 -> solve linear system for (x1,x2,x4)
            x3 = np.random.uniform(*RANGE_LINEAR)
            # simplify: sample x1, solve x2 from g, then x4 from h
            x1 = np.random.uniform(*RANGE_LINEAR)
            x2 = (z_sp - a*x1)/b
            denom = INV_DELTA
            if abs(denom) < 1e-8: continue
            x4 = (z_inv - INV_ALPHA*x1 - INV_BETA*x2 - INV_GAMMA*x3)/denom

        # Check that sampled variables are in range (solved variables can be anything)
        sampled_vars = []
        if branch == 0:  # sampled x1, x3
            sampled_vars = [x1, x3]
        elif branch == 1:  # sampled x1, x4
            sampled_vars = [x1, x4]
        elif branch == 2:  # sampled x2, x3
            sampled_vars = [x2, x3]
        elif branch == 3:  # sampled x2, x4
            sampled_vars = [x2, x4]
        elif branch == 4:  # sampled x3, x4
            sampled_vars = [x3, x4]
        elif branch == 5:  # sampled x1, x3
            sampled_vars = [x1, x3]
        elif branch == 6:  # sampled x1, x4
            sampled_vars = [x1, x4]
        else:  # branch 7, sampled x1, x3
            sampled_vars = [x1, x3]
        
        if all(RANGE_LINEAR[0] <= v <= RANGE_LINEAR[1] for v in sampled_vars):
            return x1, x2, x3, x4, z_sp, z_inv

    return None


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(n_samples_per_class=DEFAULT_SAMPLES_PER_CLASS):
    X, Y, D = [], [], []
    failures = 0

    for domain in [1, 2]:
        for y in [-1, 1]:
            collected = 0
            while collected < n_samples_per_class:
                z_sp = Z_SPU_POS if y == +1 else Z_SPU_NEG
                z_inv = Z_INV_POS if y == +1 else Z_INV_NEG
                pt = sample_point(domain,z_sp,z_inv)
                if pt is not None:
                    X.append(pt[:4])  # Take x1, x2, x3, x4
                    Y.append(y)
                    D.append(domain)
                    collected += 1
                else:
                    failures += 1

    print(f"Failed samples discarded: {failures}")
    return np.array(X), np.array(Y), np.array(D)

# ============================================================
# Interventions
# ============================================================

def make_intervention_datasets(X, Y, D):
    """
    Intervention: regenerate samples with flipped invariant target
    while keeping spurious target the same.
    """
    X_inv_int, Y_inv_int, D_inv_int = [], [], []

    for _, y, d in zip(X, Y, D):
        # Spurious target stays the same
        z_sp = Z_SPU_POS if y == +1 else Z_SPU_NEG
        # Flip invariant target
        z_inv_target = Z_INV_NEG if y == +1 else Z_INV_POS

        # Generate new point with flipped invariant
        pt = sample_point(d, z_sp, z_inv_target)
        if pt is None:
            continue

        x1, x2, x3, x4, _, _ = pt
        X_inv_int.append([x1, x2, x3, x4])
        Y_inv_int.append(y)   # class stays the same
        D_inv_int.append(d)   # domain stays the same

    return np.array(X_inv_int), np.array(Y_inv_int), np.array(D_inv_int)



# ============================================================
# Train/test preparation
# ============================================================

def prepare_data_for_training(X, Y, D, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE):
    domain1_mask = (D == 1)
    domain2_mask = (D == 2)
    
    X1, Y1 = X[domain1_mask], Y[domain1_mask]
    X2, Y2 = X[domain2_mask], Y[domain2_mask]
    
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

# ============================================================
# Data saving and loading
# ============================================================

def save_generated_data(X, Y, D, intervention_data, train_test_data):
    """Save all generated data to files"""
    os.makedirs(GENERATED_DATA_DIR, exist_ok=True)
    
    # Save raw data
    raw_data_file = os.path.join(GENERATED_DATA_DIR, RAW_DATA_FILE)
    np.savez(raw_data_file, X=X, Y=Y, D=D)
    
    # Save intervention data (only invariant intervention now)
    X_inv_int, Y_inv_int, D_inv_int = intervention_data
    intervention_data_file = os.path.join(GENERATED_DATA_DIR, INTERVENTION_DATA_FILE)
    np.savez(intervention_data_file, 
             X_inv_int=X_inv_int, Y_inv_int=Y_inv_int, D_inv_int=D_inv_int)
    
    # Save train/test split
    train_test_file = os.path.join(GENERATED_DATA_DIR, TRAIN_TEST_SPLIT_FILE)
    np.savez(train_test_file,
             X1_train=train_test_data['domain1']['train'][0], Y1_train=train_test_data['domain1']['train'][1],
             X1_test=train_test_data['domain1']['test'][0], Y1_test=train_test_data['domain1']['test'][1],
             X2_train=train_test_data['domain2']['train'][0], Y2_train=train_test_data['domain2']['train'][1],
             X2_test=train_test_data['domain2']['test'][0], Y2_test=train_test_data['domain2']['test'][1])
    
    print(f"Data saved to {GENERATED_DATA_DIR}/")

def load_generated_data():
    """Load all generated data from files"""
    # Load raw data
    raw_data_file = os.path.join(GENERATED_DATA_DIR, RAW_DATA_FILE)
    raw_data = np.load(raw_data_file)
    X, Y, D = raw_data['X'], raw_data['Y'], raw_data['D']
    
    # Load intervention data (only invariant intervention now)
    intervention_data_file = os.path.join(GENERATED_DATA_DIR, INTERVENTION_DATA_FILE)
    intervention_data = np.load(intervention_data_file)
    inv_data = (intervention_data['X_inv_int'], intervention_data['Y_inv_int'], intervention_data['D_inv_int'])
    
    # Load train/test split
    train_test_file = os.path.join(GENERATED_DATA_DIR, TRAIN_TEST_SPLIT_FILE)
    train_test_data = np.load(train_test_file)
    train_test_data = {
        'domain1': {
            'train': (train_test_data['X1_train'], train_test_data['Y1_train']),
            'test': (train_test_data['X1_test'], train_test_data['Y1_test'])
        },
        'domain2': {
            'train': (train_test_data['X2_train'], train_test_data['Y2_train']),
            'test': (train_test_data['X2_test'], train_test_data['Y2_test'])
        }
    }
    
    return X, Y, D, inv_data, train_test_data

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    X, Y, D = generate_dataset(DEFAULT_SAMPLES_PER_CLASS)
    print(f"Generated {len(X)} samples")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, D shape: {D.shape}")

    print("Creating intervention datasets...")
    intervention_data = make_intervention_datasets(X, Y, D)
    X_inv_int, Y_inv_int, D_inv_int = intervention_data
    print(f"Invariant intervention: {len(X_inv_int)} samples")

    print("Preparing training data...")
    train_test_data = prepare_data_for_training(X, Y, D)

    print("Saving data...")
    save_generated_data(X, Y, D, intervention_data, train_test_data)

    print("Data generation complete!")
