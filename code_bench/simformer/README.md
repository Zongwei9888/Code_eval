# Simformer: A Unified Transformer-based Model for Bayesian Inference

This repository contains a PyTorch implementation of the Simformer model, as described in the paper "A unified Transformer-based model for Bayesian inference". The model is capable of solving a wide variety of Bayesian inference tasks (posterior, likelihood, joint) using a single, universal architecture.

The core idea is to leverage a graph inversion algorithm to dynamically compute task-specific attention masks for a Transformer, allowing it to adapt to the conditional dependencies of any given inference problem.

## Project Structure

The codebase is organized as follows:

```
simformer/
├── model/
│   ├── transformer.py        # Core Simformer Transformer architecture
│   ├── graph.py            # Graph Inversion algorithm
│   └── tokenizer.py        # Custom tokenizer for embeddings
├── tasks/
│   ├── mask_generator.py   # Generates M_E dependency masks for all tasks
│   ├── simulators.py       # Simulators for all tasks (LV, HH, etc.)
│   └── hodgkin_huxley.py   # Specific HH energy calculation
├── inference/
│   └── sbi_wrapper.py      # Wrapper for sbi library integration
├── evaluation/
│   └── c2st.py             # C2ST metric implementation
├── configs/
│   └── default.yaml        # Hyperparameters and experiment settings
├── main.py                 # Main training and evaluation script
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd simformer
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** This project requires PyTorch, JAX, and other scientific computing libraries. If you have a CUDA-enabled GPU, ensure you install the correct version of PyTorch and JAX with GPU support for optimal performance.

## Usage

The main entry point for the project is `main.py`. It handles loading the configuration, setting up the specified tasks, training the model, and running a basic evaluation.

### Configuration

All experiment settings are managed through the `configs/default.yaml` file. This includes:
-   **Model hyperparameters** (`embedding_dim`, `num_layers`, etc.)
-   **Training parameters** (`learning_rate`, `batch_size`, `num_simulations`, etc.)
-   **Task specifications** (number of parameters and data dimensions for each task)
-   **Environment settings** (e.g., `device: 'cuda'` or `device: 'cpu'`)

You can modify this file to change the model architecture, training procedure, or the set of tasks to run.

### Running Training and Evaluation

To run the entire pipeline (training and evaluation for all tasks specified in the config), execute the `main.py` script:

```bash
python main.py
```

The script will iterate through each task defined in `configs/default.yaml`. For each task, it will:
1.  Load the corresponding simulator and dependency mask (`M_E`).
2.  Instantiate the Simformer model.
3.  Wrap the model for compatibility with the `sbi` library.
4.  Generate simulation data.
5.  Train the model to learn the posterior distribution using Sequential Neural Posterior Estimation (SNPE).
6.  Perform a simple evaluation by:
    -   Generating a synthetic observation.
    -   Drawing samples from the learned posterior.
    -   Calculating a C2ST score to measure the quality of the posterior samples against the prior.

The output will be printed to the console, showing the progress for each task and the final C2ST score.
