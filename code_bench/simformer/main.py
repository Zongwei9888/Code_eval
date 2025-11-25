"""
Main script for training and evaluating the Simformer model across various tasks.
"""
import yaml
import torch
import numpy as np
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.simulation import simulate_in_batches

from simformer.model.transformer import Simformer
from simformer.inference.sbi_wrapper import build_sbi_estimator
import simformer.tasks.mask_generator as mg
from simformer.tasks.simulators import SIMULATORS
from simformer.evaluation.c2st import c2st

def main():
    """
    Main function to orchestrate the training and evaluation pipeline.
    """
    # 1. Load Configuration
    try:
        with open("simformer/configs/default.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: `simformer/configs/default.yaml` not found. Please ensure the config file exists.")
        return

    # 2. Set up Environment
    device = config["environment"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Task-to-Mask mapping
    task_masks = {
        "gaussian": mg.M_E_gaussian,
        "two_moons": mg.M_E_two_moons,
        "gaussian_mixture": mg.M_E_two_moons,
        "slcp": mg.M_E_slcp,
        "tree": mg.M_E_tree,
        "hmm": mg.M_E_hmm,
    }
    
    # Add placeholder masks for tasks where they are not explicitly defined in the paper's addendum.
    # A fully connected graph is a safe default, allowing all possible dependencies.
    lv_dims = config['tasks']['lotka_volterra']['num_params'] + config['tasks']['lotka_volterra']['num_data']
    task_masks['lotka_volterra'] = np.ones((lv_dims, lv_dims), dtype=bool)

    hh_dims = config['tasks']['hodgkin_huxley']['num_params'] + config['tasks']['hodgkin_huxley']['num_data']
    task_masks['hodgkin_huxley'] = np.ones((hh_dims, hh_dims), dtype=bool)

    # 4. Iterate through tasks defined in the config
    for task_name, task_config in config["tasks"].items():
        print(f"\n{'='*20} Running task: {task_name.upper()} {'='*20}")

        if task_name not in task_masks:
            print(f"Skipping task '{task_name}': M_E mask not found.")
            continue

        # a. Get task-specific parameters
        num_params = task_config["num_params"]
        num_data = task_config["num_data"]
        num_vars = num_params + num_data
        m_e = task_masks[task_name]

        # b. Instantiate Model
        simformer_model = Simformer(
            m_e=m_e,
            num_vars=num_vars,
            embedding_dim=config["model"]["embedding_dim"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            d_ff=config["model"]["d_ff"],
            dropout=config["model"]["dropout"],
        ).to(device)

        # c. Get Simulator and Prior
        simulator_func = SIMULATORS[task_name]
        prior = BoxUniform(low=-3 * torch.ones(num_params, device=device), high=3 * torch.ones(num_params, device=device))

        # d. Build SBI Estimator (using NPE for posterior estimation)
        inference_method = 'npe'
        density_estimator_build_fn = build_sbi_estimator(
            simformer_model=simformer_model,
            num_params=num_params,
            num_data=num_data,
            task_type=inference_method,
            z_score_x=True,
            z_score_theta=True,
        )
        
        inference = SNPE(prior=prior, density_estimator=density_estimator_build_fn, device=device)

        # e. Generate Simulations
        print(f"Generating {config['training']['num_simulations']} simulations...")
        theta, x = simulate_in_batches(
            simulator=simulator_func,
            proposal=prior,
            num_simulations=config["training"]["num_simulations"],
            simulation_batch_size=config["training"]["batch_size"],
        )

        # f. Train the estimator
        print("Training the density estimator...")
        density_estimator = inference.append_simulations(theta, x).train(
            training_batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            max_num_epochs=config["training"]["num_epochs"],
            show_train_summary=False,
            stop_after_epochs=5 # Use early stopping for efficient demonstration
        )

        # g. Evaluation
        print("Performing evaluation...")
        
        # Generate a synthetic observation x_0 to condition the posterior on.
        true_theta = prior.sample((1,))
        x_0 = simulator_func(true_theta)
        
        posterior = inference.build_posterior()

        print(f"Drawing {config['training']['num_posterior_samples']} posterior samples...")
        posterior_samples = posterior.sample((config["training"]["num_posterior_samples"],), x=x_0)

        # As a proxy for a ground-truth posterior, we compare posterior samples against prior samples.
        # A good posterior should be distinguishable from the prior (C2ST score -> 1.0).
        prior_samples = prior.sample((config["training"]["num_posterior_samples"],))
        
        X = posterior_samples.detach().cpu().numpy()
        Y = prior_samples.detach().cpu().numpy()

        if X.ndim == 1: X = X.reshape(-1, 1)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)

        if X.shape[0] > 0 and Y.shape[0] > 0:
            print("Calculating C2ST score (Posterior vs. Prior)...")
            score = c2st(X, Y, n_folds=config["evaluation"]["c2st_folds"])
            print(f"C2ST Score: {score:.4f}")
        else:
            print("Warning: Could not generate enough samples for C2ST evaluation.")

        print(f"--- Task {task_name.upper()} finished ---")

if __name__ == "__main__":
    main()
