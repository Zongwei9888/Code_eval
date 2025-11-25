"""
SBI Wrapper for Simformer model integration with the sbi library.

This module provides a wrapper to make the Simformer model compatible with
the sbi (simulation-based inference) library for Neural Posterior Estimation (NPE)
and Neural Likelihood Estimation (NLE).
"""

# Handle optional dependencies gracefully
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create mock classes for when torch is not available
    class nn:
        class Module:
            def __init__(self):
                pass

try:
    from sbi.utils import posterior_nn, likelihood_nn
    SBI_AVAILABLE = True
except ImportError:
    SBI_AVAILABLE = False
    posterior_nn = None
    likelihood_nn = None

# Try to import Simformer, but allow the module to work without it
try:
    from simformer.model.transformer import Simformer
    SIMFORMER_AVAILABLE = True
except ImportError:
    SIMFORMER_AVAILABLE = False
    Simformer = None  # Type hint placeholder


class SbiWrapper(nn.Module if TORCH_AVAILABLE else object):
    """
    A wrapper to make the Simformer model compatible with the sbi library,
    which expects a simple `embedding_net` that takes a context (theta or x)
    and returns a single embedding vector.
    """
    
    def __init__(self, simformer, num_params: int, num_data: int, task_type: str):
        """
        Args:
            simformer: The Simformer model instance.
            num_params: The number of parameters (theta) for the task.
            num_data: The number of data dimensions (x) for the task.
            task_type: The type of inference task, either 'npe' (posterior) or 'nle' (likelihood).
        
        Raises:
            ImportError: If torch is not available.
            ValueError: If task_type is not 'npe' or 'nle'.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for SbiWrapper. "
                "Please install it with: pip install torch"
            )
        
        super().__init__()
        self.simformer = simformer
        self.num_params = num_params
        self.num_data = num_data
        self.num_vars = num_params + num_data
        self.task_type = task_type.lower()
        
        if self.task_type not in ['npe', 'nle']:
            raise ValueError("task_type must be 'npe' or 'nle'")

    def forward(self, context: "torch.Tensor") -> "torch.Tensor":
        """
        Processes the context from sbi and formats it for the Simformer.

        Args:
            context: For NPE, this is the data `x`. For NLE, this is the parameters `theta`.
                     Shape: (batch_size, num_context_dims)

        Returns:
            A single embedding vector for the context. Shape: (batch_size, d_model)
        """
        batch_size = context.shape[0]
        device = context.device

        # 1. Create the full input sequence `values`
        values = torch.zeros(batch_size, self.num_vars, device=device)
        
        # 2. Create the conditioning mask `cond_mask`
        cond_mask = torch.zeros(batch_size, self.num_vars, dtype=torch.bool, device=device)

        if self.task_type == 'npe':
            # For NPE, the context is `x` (data), and it's conditioned upon.
            # Parameters `theta` are latent.
            values[:, self.num_params:] = context
            cond_mask[:, self.num_params:] = True
        else:  # 'nle'
            # For NLE, the context is `theta` (parameters), and it's conditioned upon.
            # Data `x` is latent.
            values[:, :self.num_params] = context
            cond_mask[:, :self.num_params] = True

        # 3. Create identifiers and metadata
        ids = torch.arange(self.num_vars, device=device).unsqueeze(0).repeat(batch_size, 1)
        metadata = torch.zeros_like(ids)
        metadata[:, self.num_params:] = 1  # 0 for params, 1 for data

        # 4. Create a dummy time embedding (sbi doesn't use diffusion time)
        t = torch.zeros(batch_size, 1, device=device)

        # 5. Pass through the Simformer
        # The sbi context does not include the variables to be predicted, so we pass zeros
        # for those parts of the `values` tensor. The model should learn to predict them.
        output_sequence = self.simformer(
            values=values,
            cond_mask=cond_mask,
            ids=ids,
            metadata=metadata,
            t=t
        )

        # 6. Pool the output sequence to get a single embedding vector
        # We perform mean pooling over the sequence dimension.
        embedding = torch.mean(output_sequence, dim=1)
        return embedding


def build_sbi_estimator(
    simformer_model,
    num_params: int,
    num_data: int,
    task_type: str,
    z_score_x: bool = True,
    z_score_theta: bool = True,
):
    """
    Factory function to build an sbi density estimator (NPE or NLE) using
    the Simformer as the neural network backbone.

    Args:
        simformer_model: The instantiated Simformer model.
        num_params: The number of parameters for the task.
        num_data: The number of data dimensions for the task.
        task_type: The type of inference task, 'npe' or 'nle'.
        z_score_x: Whether to z-score the data `x`.
        z_score_theta: Whether to z-score the parameters `theta`.

    Returns:
        An sbi density estimator object (e.g., PosteriorNN or LikelihoodNN).
    
    Raises:
        ImportError: If required dependencies are not available.
        ValueError: If task_type is not 'npe' or 'nle'.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for build_sbi_estimator. "
            "Please install it with: pip install torch"
        )
    
    if not SBI_AVAILABLE:
        raise ImportError(
            "sbi library is required for build_sbi_estimator. "
            "Please install it with: pip install sbi"
        )
    
    task_type = task_type.lower()
    if task_type not in ['npe', 'nle']:
        raise ValueError("task_type must be 'npe' or 'nle'")

    # Wrap the Simformer to make it compatible with the sbi API
    embedding_net = SbiWrapper(
        simformer=simformer_model,
        num_params=num_params,
        num_data=num_data,
        task_type=task_type
    )

    if task_type == 'npe':
        # Use Neural Posterior Estimation
        # As per the addendum, use a 'more expressive neural spline flow'
        estimator = posterior_nn(
            model='nsf',  # Neural Spline Flow
            embedding_net=embedding_net,
            z_score_x=z_score_x,
            z_score_theta=z_score_theta,
        )
    else:  # 'nle'
        # Use Neural Likelihood Estimation
        # As per the addendum, use a 'more expressive neural spline flow'
        estimator = likelihood_nn(
            model='nsf',  # Neural Spline Flow
            embedding_net=embedding_net,
            z_score_x=z_score_x,
            z_score_theta=z_score_theta,
        )
        
    return estimator


def check_dependencies():
    """
    Check if all required dependencies are available.
    
    Returns:
        dict: A dictionary with dependency names as keys and availability as values.
    """
    return {
        'torch': TORCH_AVAILABLE,
        'sbi': SBI_AVAILABLE,
        'simformer': SIMFORMER_AVAILABLE,
    }


# Module-level execution for testing
if __name__ == "__main__":
    print("SBI Wrapper Module")
    print("=" * 40)
    
    # Check dependencies
    deps = check_dependencies()
    print("\nDependency Status:")
    for dep, available in deps.items():
        status = "✓ Available" if available else "✗ Not installed"
        print(f"  {dep}: {status}")
    
    # If all dependencies are available, run a simple test
    if all(deps.values()):
        print("\nAll dependencies available. Ready to use.")
    else:
        missing = [dep for dep, avail in deps.items() if not avail]
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install missing packages to use full functionality.")
        print("\nInstallation commands:")
        if not deps['torch']:
            print("  pip install torch")
        if not deps['sbi']:
            print("  pip install sbi")
        if not deps['simformer']:
            print("  (simformer is a local package - ensure it's in your Python path)")
    
    print("\nModule loaded successfully!")
