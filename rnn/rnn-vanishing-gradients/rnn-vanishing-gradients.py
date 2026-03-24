import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    rho = np.linalg.norm(W_hh, ord=2)  # ||W_hh||₂
    
    # Step 2: Start with gradient norm = 1.0 at last step
    gradient_norms = [1.0]
    
    # Step 3: For each step back in time, multiply by shrink factor
    for t in range(1, T):
        current_norm = gradient_norms[-1] * rho  # previous × shrink_factor
        gradient_norms.append(current_norm)
    
    return gradient_norms

    pass