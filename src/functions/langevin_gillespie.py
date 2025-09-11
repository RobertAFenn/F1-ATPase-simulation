import numpy as np
import math

"""
Simulate Langevin dynamics using the Langevin-Gillespie algorithm for multiple independent simulations.

This function implements stochastic simulations of 1D Langevin dynamics with different integration methods.
It tracks the position, discrete state, and reference angle (θ) over time for each simulation.

Parameters
----------
nSim : int
    Number of independent simulations to run.
dur : int
    Number of time steps in each simulation.
dt : float
    Size of each time step.
pos : array_like, shape (nSim,)
    Initial positions for each simulation.
gammaB : float
    Rotational friction coefficient, affecting how quickly the system changes orientation.
kappa : float
    Elastic constant, used in the drift term that influences particle movement.
kBT : float
    Thermal energy (Boltzmann constant * temperature), used in the stochastic diffusion term.
method : str
    Integration method to use. Options are:
        - 'heun'         : Heun's method (predictor-corrector)
        - 'euler'        : Euler-Maruyama method
        - 'probabilistic': Probabilistic integration for stochastic jumps

Returns
-------
tuple of np.ndarray
    Returns three arrays of shape (dur, nSim):
    pos_store   : Positions over time for each simulation.
    state_store : Discrete states over time for each simulation.
    theta_store : Reference angles (θ) over time for each simulation.

Notes
-----
- The initial state is assumed to be 0 (inactive) and the initial reference angle θ is 0 by default.
- The function uses NumPy for array operations and random number generation, and math for scalar operations.
- Currently, `state` and `theta_i` remain constant over time; update logic should be added if they are dynamic.
"""


def langevin_gillespie(
    nSim,  # number of simulations
    dur,  # number of time steps
    dt,  # time step size
    pos,  # initial positions for each simulation, array of length nSim
    gammaB,  # rotational friction
    kappa,  # elastic constant
    kBT,  # thermal energy
    method,  # integration method: 'heun', 'euler', 'probabilistic'
):
    # Parameter Pre-Processing
    method = method.lower()  # make method lowercase so string matching works

    # Preallocate storage arrays: shape (dur, nSim)
    pos_store = np.full((dur, nSim), np.nan)  # [i, k] => position at time step i for simulation k
    state_store = np.full((dur, nSim), np.nan)  # [i, k] => discrete state at time step i for simulation k
    theta_store = np.full((dur, nSim), np.nan)  # [i, k] => reference angle (θ) at time step i for simulation k

    theta_i = 0
    state = 0  # initial state
    for k in range(nSim):  # simulations
        pos_extend, pos_store[0, k] = pos[k], pos[k]  # pos_extend evolves over time due to dt
        state_store[0, k] = 1 if state == 1 else 0
        theta_store[0, k] = theta_i

        for i in range(1, dur):  # time steps (start at 1, as 0 is the initial position)
            match method:  # Every case should be lowercase
                case "heun":
                    pos_extend = Heun1D(pos_extend, dt, gammaB, kappa, theta_i, kBT)

                case "euler":
                    pos_extend = computeEulerMaruyama(pos_extend, dt, gammaB, kappa, theta_i, kBT)

                case "probabilistic":
                    pos_extend = computeProbabilistic(pos_extend, dt, gammaB, kappa, theta_i, kBT)

                case _:
                    raise ValueError("ERROR: Method must be a string and defined as one of the following (heun, euler, probabilistic)")

            # Store results for this step
            pos_store[i, k] = pos_extend
            state_store[i, k] = 1 if state == 1 else 0
            theta_store[i, k] = theta_i

    return pos_store, state_store, theta_store


# Helper Functions
def drift(x, gammaB, kappa, theta_i):
    return 1 / gammaB * (-kappa * (x - theta_i))


def diffusion(gammaB, kBT):
    return 1 / gammaB * math.sqrt(2 * gammaB * kBT)


def Heun1D(pos, dt, gammaB, kappa, theta_i, kBT):
    driftX = drift(pos, gammaB, kappa, theta_i)
    diffusionX = diffusion(gammaB, kBT)
    eta = np.random.randn()  # standard normal

    y_predict = pos + dt * driftX + math.sqrt(dt) * diffusionX * eta  # Predictor step
    drift_predict = drift(y_predict, gammaB, kappa, theta_i)  # Corrector step

    pos_next = pos + (dt / 2) * (driftX + drift_predict) + math.sqrt(dt) * diffusionX * eta

    return pos_next


def computeEulerMaruyama(pos, dt, gammaB, kappa, theta_i, kBT):
    return pos + dt * (1 / gammaB) * (-kappa * (pos - theta_i)) + math.sqrt(2 * kBT / gammaB * dt) * np.random.randn()


def computeProbabilistic(pos, dt, gammaB, kappa, theta_i, kBT):
    return pos + dt * (1 / gammaB) * (-kappa * (pos - theta_i)) + math.sqrt(2 * kBT / gammaB * dt) * np.random.randn()
