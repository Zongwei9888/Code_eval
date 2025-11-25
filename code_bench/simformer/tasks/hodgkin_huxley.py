# Fixed: Replaced jax.numpy with numpy for broader compatibility
import numpy as np

def convert_charge_to_energy(E):
    """
    Computes the energy consumption summary statistic for the Hodgkin-Huxley task.

    This function implements the JAX-based energy calculation provided in the
    paper's addendum.

    Args:
        E: Charge data from the Hodgkin-Huxley simulation.

    Returns:
        Energy consumption in microjoules per second.
    """
    E = np.diff(E)
    E = np.convolve(E, 1/5*np.ones(5), mode="same")
    E = -E / 1000 / 1000 * 0.628e-3
    e = 1.602176634e-19
    N_Na = E / e
    ATP_Na = N_Na / (1 * 3)
    ATP_energy = 10e-19
    E_joules = ATP_Na * ATP_energy
    E_joules_per_sec = E_joules / 0.2
    return E_joules_per_sec * 1e+6
