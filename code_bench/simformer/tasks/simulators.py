"""
This module provides a collection of simulator functions for various tasks,
designed to be compatible with the sbi library. Each simulator takes a tensor
of parameters and returns a tensor of simulated data.

This module gracefully handles missing dependencies (torch, sbi) by providing
fallback implementations using numpy or pure Python.
"""
import math
import random
from typing import Union, List

# Try to import torch, fall back to numpy, or use pure Python
try:
    import torch
    BACKEND = "torch"
except ImportError:
    try:
        import numpy as np
        BACKEND = "numpy"
    except ImportError:
        BACKEND = "python"
        np = None

# Try to import sbi simulators (optional)
try:
    from sbi.simulators import (
        linear_gaussian,
        two_moons,
        lotka_volterra,
        gaussian_mixture,
    )
    SBI_AVAILABLE = True
except ImportError:
    SBI_AVAILABLE = False
    linear_gaussian = None
    two_moons = None
    lotka_volterra = None
    gaussian_mixture = None


# ============================================================================
# Backend-agnostic helper functions
# ============================================================================

def _ensure_2d(theta):
    """Ensure theta is 2D (batch_size, num_params)."""
    if BACKEND == "torch":
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
    elif BACKEND == "numpy":
        if theta.ndim == 1:
            theta = np.expand_dims(theta, 0)
    else:  # pure python - assume list of lists or single list
        if not isinstance(theta[0], (list, tuple)):
            theta = [theta]
    return theta


def _randn(shape):
    """Generate random normal samples."""
    if BACKEND == "torch":
        return torch.randn(shape)
    elif BACKEND == "numpy":
        return np.random.randn(*shape)
    else:
        # Pure Python implementation
        if isinstance(shape, int):
            shape = (shape,)
        total = 1
        for s in shape:
            total *= s
        # Box-Muller transform for normal distribution
        samples = []
        for _ in range(total):
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1 + 1e-10)) * math.cos(2 * math.pi * u2)
            samples.append(z)
        # Reshape to desired shape
        return _reshape_list(samples, shape)


def _randn_like(x):
    """Generate random normal samples with same shape as x."""
    if BACKEND == "torch":
        return torch.randn_like(x)
    elif BACKEND == "numpy":
        return np.random.randn(*x.shape)
    else:
        if isinstance(x, list):
            return [random.gauss(0, 1) for _ in x]
        return random.gauss(0, 1)


def _reshape_list(flat_list, shape):
    """Reshape a flat list into nested lists of given shape."""
    if len(shape) == 1:
        return flat_list[:shape[0]]
    result = []
    stride = 1
    for s in shape[1:]:
        stride *= s
    for i in range(shape[0]):
        result.append(_reshape_list(flat_list[i*stride:(i+1)*stride], shape[1:]))
    return result


def _sin(x):
    """Element-wise sine."""
    if BACKEND == "torch":
        return torch.sin(x)
    elif BACKEND == "numpy":
        return np.sin(x)
    else:
        if isinstance(x, list):
            return [math.sin(v) for v in x]
        return math.sin(x)


def _cos(x):
    """Element-wise cosine."""
    if BACKEND == "torch":
        return torch.cos(x)
    elif BACKEND == "numpy":
        return np.cos(x)
    else:
        if isinstance(x, list):
            return [math.cos(v) for v in x]
        return math.cos(x)


def _stack(arrays, axis=1):
    """Stack arrays along an axis."""
    if BACKEND == "torch":
        return torch.stack(arrays, dim=axis)
    elif BACKEND == "numpy":
        return np.stack(arrays, axis=axis)
    else:
        # For pure Python, assume axis=1 and arrays are 1D lists
        return [[arr[i] if isinstance(arr, list) else arr for arr in arrays] 
                for i in range(len(arrays[0]) if isinstance(arrays[0], list) else 1)]


def _cat(tensors, axis=1):
    """Concatenate tensors along an axis."""
    if BACKEND == "torch":
        return torch.cat(tensors, dim=axis)
    elif BACKEND == "numpy":
        return np.concatenate(tensors, axis=axis)
    else:
        # Pure Python concatenation along axis 1
        result = []
        num_rows = len(tensors[0])
        for i in range(num_rows):
            row = []
            for t in tensors:
                if isinstance(t[i], list):
                    row.extend(t[i])
                else:
                    row.append(t[i])
            result.append(row)
        return result


def _add(a, b):
    """Element-wise addition."""
    if BACKEND in ("torch", "numpy"):
        return a + b
    else:
        if isinstance(a, list) and isinstance(b, list):
            if isinstance(a[0], list):
                return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(len(a))]
            return [a[i] + b[i] for i in range(len(a))]
        return a + b


def _get_shape(x):
    """Get shape of array/tensor."""
    if BACKEND == "torch":
        return tuple(x.shape)
    elif BACKEND == "numpy":
        return x.shape
    else:
        if isinstance(x, list):
            if isinstance(x[0], list):
                return (len(x), len(x[0]))
            return (len(x),)
        return ()


# ============================================================================
# Simulator Functions
# ============================================================================

def gaussian_linear_simulator(theta) -> Union['torch.Tensor', 'np.ndarray', list]:
    """
    Simulates a 10x10 linear Gaussian model.
    - 10 parameters (theta)
    - 10 data points (x)
    
    Falls back to a simple implementation if sbi is not available.
    """
    theta = _ensure_2d(theta)
    
    if SBI_AVAILABLE and BACKEND == "torch":
        # Use sbi's linear_gaussian if available
        return linear_gaussian(theta, num_dim=10)
    else:
        # Fallback: x = theta + noise (simple linear Gaussian)
        shape = _get_shape(theta)
        num_samples = shape[0]
        noise = _randn((num_samples, 10))
        return _add(theta, noise)


def two_moons_simulator(theta) -> Union['torch.Tensor', 'np.ndarray', list]:
    """
    Simulates the Two Moons dataset.
    - 2 parameters (theta)
    - 2 data points (x)
    
    Falls back to a simple implementation if sbi is not available.
    """
    theta = _ensure_2d(theta)
    
    if SBI_AVAILABLE and BACKEND == "torch":
        return two_moons(theta)
    else:
        # Fallback: simplified two moons simulation
        shape = _get_shape(theta)
        num_samples = shape[0]
        noise = _randn((num_samples, 2))
        # Simple nonlinear transformation
        if BACKEND == "numpy":
            x = np.column_stack([
                np.sin(theta[:, 0]) + 0.1 * noise[:, 0],
                np.cos(theta[:, 1]) + 0.1 * noise[:, 1]
            ])
        elif BACKEND == "torch":
            x = torch.stack([
                torch.sin(theta[:, 0]) + 0.1 * noise[:, 0],
                torch.cos(theta[:, 1]) + 0.1 * noise[:, 1]
            ], dim=1)
        else:
            x = [[math.sin(theta[i][0]) + 0.1 * noise[i][0],
                  math.cos(theta[i][1]) + 0.1 * noise[i][1]] 
                 for i in range(num_samples)]
        return x


def gaussian_mixture_simulator(theta) -> Union['torch.Tensor', 'np.ndarray', list]:
    """
    Simulates a Gaussian Mixture Model.
    - 2 parameters (theta)
    - 2 data points (x)
    
    Falls back to a simple implementation if sbi is not available.
    """
    theta = _ensure_2d(theta)
    
    if SBI_AVAILABLE and BACKEND == "torch":
        return gaussian_mixture(theta)
    else:
        # Fallback: simple Gaussian mixture simulation
        shape = _get_shape(theta)
        num_samples = shape[0]
        noise = _randn((num_samples, 2))
        # Add noise to theta
        return _add(theta, noise)


def slcp_simulator(theta) -> Union['torch.Tensor', 'np.ndarray', list]:
    """
    Custom simulator for the Simple Likelihood, Complex Posterior (SLCP) task.
    - 4 parameters (theta)
    - 8 data points (x)
    The dependency mask suggests pairs of data, where each pair depends on all parameters.
    x_i ~ N(theta, I)
    """
    theta = _ensure_2d(theta)
    shape = _get_shape(theta)
    num_samples = shape[0]
    num_dim = 8
    
    if BACKEND == "torch":
        # Original torch implementation
        mean = torch.cat([
            torch.sin(theta[:, :2]),
            torch.cos(theta[:, 2:]),
            torch.sin(theta[:, :1] + theta[:, 1:2]),
            torch.cos(theta[:, 2:3] + theta[:, 3:4]),
            torch.sin(theta[:, 1:2] + theta[:, 2:3]),
            torch.cos(theta[:, :1] + theta[:, 3:4]),
            torch.sin(theta[:, :2].sum(dim=1, keepdim=True)),
            torch.cos(theta[:, 2:].sum(dim=1, keepdim=True)),
        ], dim=1)
        x = torch.randn(num_samples, num_dim) + mean
    elif BACKEND == "numpy":
        mean = np.concatenate([
            np.sin(theta[:, :2]),
            np.cos(theta[:, 2:]),
            np.sin(theta[:, :1] + theta[:, 1:2]),
            np.cos(theta[:, 2:3] + theta[:, 3:4]),
            np.sin(theta[:, 1:2] + theta[:, 2:3]),
            np.cos(theta[:, :1] + theta[:, 3:4]),
            np.sin(theta[:, :2].sum(axis=1, keepdims=True)),
            np.cos(theta[:, 2:].sum(axis=1, keepdims=True)),
        ], axis=1)
        x = np.random.randn(num_samples, num_dim) + mean
    else:
        # Pure Python implementation
        x = []
        for i in range(num_samples):
            t = theta[i]
            mean = [
                math.sin(t[0]), math.sin(t[1]),
                math.cos(t[2]), math.cos(t[3]),
                math.sin(t[0] + t[1]),
                math.cos(t[2] + t[3]),
                math.sin(t[1] + t[2]),
                math.cos(t[0] + t[3]),
            ]
            row = [mean[j] + random.gauss(0, 1) for j in range(num_dim)]
            x.append(row)
    return x


def tree_simulator(theta) -> Union['torch.Tensor', 'np.ndarray', list]:
    """
    Custom simulator for a 10-variable tree-structured Bayesian network.
    Based on the mask: M_E_tree[0, 1:3] = True, M_E_tree[1, 3:5] = True, M_E_tree[2, 5:7] = True
    This implies:
    - v0 depends on v1, v2
    - v1 depends on v3, v4
    - v2 depends on v5, v6
    - v3, v4, v5, v6, v7, v8, v9 are root nodes (parameters).
    So, 7 parameters and 3 data points.
    """
    theta = _ensure_2d(theta)
    
    if BACKEND == "torch":
        v3, v4, v5, v6 = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
        v1 = 0.5 * (v3 + v4) + 0.1 * torch.randn_like(v3)
        v2 = 0.5 * (v5 + v6) + 0.1 * torch.randn_like(v5)
        v0 = 0.5 * (v1 + v2) + 0.1 * torch.randn_like(v1)
        x = torch.stack([v0, v1, v2], dim=1)
    elif BACKEND == "numpy":
        v3, v4, v5, v6 = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
        v1 = 0.5 * (v3 + v4) + 0.1 * np.random.randn(*v3.shape)
        v2 = 0.5 * (v5 + v6) + 0.1 * np.random.randn(*v5.shape)
        v0 = 0.5 * (v1 + v2) + 0.1 * np.random.randn(*v1.shape)
        x = np.stack([v0, v1, v2], axis=1)
    else:
        # Pure Python
        x = []
        for t in theta:
            v3, v4, v5, v6 = t[0], t[1], t[2], t[3]
            v1 = 0.5 * (v3 + v4) + 0.1 * random.gauss(0, 1)
            v2 = 0.5 * (v5 + v6) + 0.1 * random.gauss(0, 1)
            v0 = 0.5 * (v1 + v2) + 0.1 * random.gauss(0, 1)
            x.append([v0, v1, v2])
    return x


def hmm_simulator(theta) -> Union['torch.Tensor', 'np.ndarray', list]:
    """
    Simulator for a Hidden Markov Model (HMM).
    - 10 parameters (latent states theta)
    - 10 data points (observations x)
    The mask M_E_hmm implies x_i depends on theta_i, and theta_i depends on theta_{i-1}.
    The simulator is just the observation model: x_i = f(theta_i).
    The transition model (theta_i ~ f(theta_{i-1})) is implicit in the prior.
    """
    theta = _ensure_2d(theta)
    
    if BACKEND == "torch":
        x = theta + 0.1 * torch.randn_like(theta)
    elif BACKEND == "numpy":
        x = theta + 0.1 * np.random.randn(*theta.shape)
    else:
        x = [[t[j] + 0.1 * random.gauss(0, 1) for j in range(len(t))] for t in theta]
    return x


def lotka_volterra_simulator(theta) -> Union['torch.Tensor', 'np.ndarray', list]:
    """
    Simulates the Lotka-Volterra model.
    - 4 parameters (theta)
    - 21*2=42 data points (x), but sbi summarizes it to 18.
    
    Falls back to a simplified implementation if sbi is not available.
    """
    theta = _ensure_2d(theta)
    
    if SBI_AVAILABLE and BACKEND == "torch":
        return lotka_volterra(theta)
    else:
        # Fallback: simplified Lotka-Volterra simulation
        # Returns 18 summary statistics as random values influenced by theta
        shape = _get_shape(theta)
        num_samples = shape[0]
        
        if BACKEND == "numpy":
            # Simple simulation based on theta parameters
            noise = np.random.randn(num_samples, 18)
            # Scale by theta values to create some dependency
            x = noise * np.abs(theta[:, :1]) + theta[:, 1:2]
        elif BACKEND == "torch":
            noise = torch.randn(num_samples, 18)
            x = noise * torch.abs(theta[:, :1]) + theta[:, 1:2]
        else:
            x = []
            for t in theta:
                row = [random.gauss(0, 1) * abs(t[0]) + t[1] for _ in range(18)]
                x.append(row)
        return x


def hodgkin_huxley_simulator(theta) -> Union['torch.Tensor', 'np.ndarray', list]:
    """
    Placeholder simulator for the Hodgkin-Huxley model.
    The real simulator is complex and requires a dedicated solver.
    The paper uses 8 summary statistics for the output.
    This function returns random data of the correct shape.
    """
    theta = _ensure_2d(theta)
    shape = _get_shape(theta)
    num_samples = shape[0]
    
    # Return 8 summary statistics
    return _randn((num_samples, 8))


# Dictionary to easily access simulators by name
SIMULATORS = {
    "gaussian": gaussian_linear_simulator,
    "two_moons": two_moons_simulator,
    "gaussian_mixture": gaussian_mixture_simulator,
    "slcp": slcp_simulator,
    "tree": tree_simulator,
    "hmm": hmm_simulator,
    "lotka_volterra": lotka_volterra_simulator,
    "hodgkin_huxley": hodgkin_huxley_simulator,
}


# ============================================================================
# Module self-test
# ============================================================================

if __name__ == "__main__":
    print(f"Backend: {BACKEND}")
    print(f"SBI Available: {SBI_AVAILABLE}")
    print()
    
    # Test each simulator with sample inputs
    test_cases = {
        "gaussian": [0.1] * 10,
        "two_moons": [0.5, -0.5],
        "gaussian_mixture": [0.3, 0.7],
        "slcp": [0.1, 0.2, 0.3, 0.4],
        "tree": [0.1, 0.2, 0.3, 0.4],
        "hmm": [0.1] * 10,
        "lotka_volterra": [0.5, 0.5, 0.5, 0.5],
        "hodgkin_huxley": [0.1] * 8,
    }
    
    print("Testing all simulators:")
    print("-" * 50)
    
    for name, theta in test_cases.items():
        try:
            simulator = SIMULATORS[name]
            result = simulator(theta)
            
            # Get result shape
            if BACKEND == "torch":
                result_shape = tuple(result.shape)
            elif BACKEND == "numpy":
                result_shape = result.shape
            else:
                if isinstance(result, list) and isinstance(result[0], list):
                    result_shape = (len(result), len(result[0]))
                elif isinstance(result, list):
                    result_shape = (len(result),)
                else:
                    result_shape = ()
            
            print(f"  {name}: OK - output shape {result_shape}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    
    print("-" * 50)
    print("All tests completed!")
