"""
Tokenizer module for converting variables, metadata, and conditioning masks into embeddings.
"""

# Attempt to import PyTorch, provide fallback if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create minimal stubs for type hints and basic functionality when torch is unavailable
    class nn:
        class Module:
            def __init__(self):
                pass
            def __call__(self, *args, **kwargs):
                raise RuntimeError("PyTorch is required but not installed. Install with: pip install torch")
        class Embedding:
            def __init__(self, *args, **kwargs):
                pass
        class Parameter:
            def __init__(self, *args, **kwargs):
                pass
    
    class torch:
        class Tensor:
            pass
        @staticmethod
        def randn(*args, **kwargs):
            return None
        @staticmethod
        def cat(*args, **kwargs):
            return None


class Tokenizer(nn.Module):
    """
    A tokenizer that converts variables, metadata, and conditioning masks into a sequence of embeddings.
    The final embedding for each variable is a concatenation of:
    [identifier_embedding, value_embedding, metadata_embedding, condition_mask_embedding].
    """
    def __init__(self, num_vars: int, embedding_dim: int):
        """
        Args:
            num_vars: The total number of variables (parameters + data).
            embedding_dim: The base dimension for each embedding component. The final
                         concatenated embedding will have dimension 4 * embedding_dim.
        
        Raises:
            RuntimeError: If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for Tokenizer but not installed. "
                "Please install it with: pip install torch"
            )
        
        super().__init__()
        self.num_vars = num_vars
        self.embedding_dim = embedding_dim

        # Learnable embedding for variable identifiers (0, 1, ..., num_vars-1)
        self.identifier_embedding = nn.Embedding(num_vars, embedding_dim)

        # Learnable embedding for metadata (e.g., 0 for parameters, 1 for data)
        self.metadata_embedding = nn.Embedding(2, embedding_dim)

        # A single learnable vector for the 'True' condition mask. 'False' will be a zero vector.
        self.condition_mask_embedding_true = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, values: "torch.Tensor", cond_mask: "torch.Tensor", ids: "torch.Tensor", metadata: "torch.Tensor") -> "torch.Tensor":
        """
        Converts raw input tensors into a sequence of concatenated embeddings.

        Args:
            values: Scalar values of the variables. Shape: (batch_size, num_vars)
            cond_mask: Conditioning mask (True if conditioned, False otherwise). Shape: (batch_size, num_vars)
            ids: Identifier for each variable (0 to num_vars-1). Shape: (batch_size, num_vars)
            metadata: Metadata for each variable (e.g., 0 for param, 1 for data). Shape: (batch_size, num_vars)

        Returns:
            A tensor of concatenated embeddings. Shape: (batch_size, num_vars, 4 * embedding_dim)
        """
        # 1. Value Embedding: Repeat the scalar value to match the embedding dimension.
        # Input shape: (batch_size, num_vars) -> Output shape: (batch_size, num_vars, embedding_dim)
        value_emb = values.unsqueeze(-1).repeat(1, 1, self.embedding_dim)

        # 2. Identifier Embedding: Look up the embedding for each variable's ID.
        # Input shape: (batch_size, num_vars) -> Output shape: (batch_size, num_vars, embedding_dim)
        id_emb = self.identifier_embedding(ids)

        # 3. Metadata Embedding: Look up the embedding for each variable's type.
        # Input shape: (batch_size, num_vars) -> Output shape: (batch_size, num_vars, embedding_dim)
        meta_emb = self.metadata_embedding(metadata)

        # 4. Condition Mask Embedding: Use the learnable 'True' vector or a zero vector.
        # We achieve this by multiplying the boolean mask with the learnable vector.
        # Convert boolean mask to float for multiplication
        # Input shape: (batch_size, num_vars) -> Output shape: (batch_size, num_vars, embedding_dim)
        cond_mask_float = cond_mask.float() if cond_mask.dtype == torch.bool else cond_mask
        cond_mask_emb = cond_mask_float.unsqueeze(-1) * self.condition_mask_embedding_true.view(1, 1, -1)

        # Concatenate all four embeddings along the last dimension.
        final_embedding = torch.cat([value_emb, id_emb, meta_emb, cond_mask_emb], dim=-1)

        return final_embedding


# Module-level test that runs when file is executed directly
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("WARNING: PyTorch is not installed.")
        print("To use this module, install PyTorch with: pip install torch")
        print("\nModule loaded successfully (in stub mode).")
    else:
        print("Testing Tokenizer class...")
        
        # Create a tokenizer instance
        num_vars = 5
        embedding_dim = 8
        tokenizer = Tokenizer(num_vars, embedding_dim)
        
        print(f"Created tokenizer with {num_vars} variables and embedding dimension {embedding_dim}")
        print(f"Expected output dimension: {4 * embedding_dim}")
        
        # Create test inputs
        batch_size = 2
        values = torch.randn(batch_size, num_vars)
        cond_mask = torch.randint(0, 2, (batch_size, num_vars)).bool()
        ids = torch.arange(num_vars).unsqueeze(0).repeat(batch_size, 1)
        metadata = torch.randint(0, 2, (batch_size, num_vars))
        
        print(f"\nInput shapes:")
        print(f"  values: {values.shape}")
        print(f"  cond_mask: {cond_mask.shape}")
        print(f"  ids: {ids.shape}")
        print(f"  metadata: {metadata.shape}")
        
        # Test forward pass
        output = tokenizer(values, cond_mask, ids, metadata)
        print(f"\nOutput shape: {output.shape}")
        print("\n✓ Tokenizer executed successfully!")
