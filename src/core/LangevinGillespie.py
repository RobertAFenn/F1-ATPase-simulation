import math
import numpy as np  # type: ignore
import random
import warnings


class LangevinGillespie:
    def __init__(self):
        # Simulation Parameters
        self.steps = None  # Number of simulation time steps
        self.dt = None  # Time step size
        self.method = None  # Integration method: "heun", "euler", or "probabilistic"
        # Physical System Parameters
        self.theta_0 = None  # Initial bead position (angle in radians)
        self.kappa = None  # Elastic constant of the system
        self.kBT = None  # Thermal energy (k_B * T)
        self.gammaB = None  # Rotational friction coefficient of the bead
        # Multi-state parameters
        self.transition_matrix = None  # Transition rate matrix between states
        self.initial_state = None  # Index of starting state
        self.theta_states = None  # Target angles for each state (in radians)

    def simulate(self, rng=None) -> tuple:
        """Run a Langevin simulation.

        Returns:
            bead_positions: array of bead angles over time
            states: array of discrete states over time
            target_thetas: array of target angles for each step
        """
        # Step -1: Check that all required attributes have been set
        self._verify_attributes()

        # Step 0: Set up random number generator
        if isinstance(rng, int):
            rand_gen = random.Random(rng)  # Use seed for reproducibility
        else:
            rand_gen = random.Random()  # Default RNG

        method_lowered = self.method.lower()

        # Step 1: Determine starting angle (local variable, does not overwrite self.theta_0)
        theta_start = self.theta_0 if self.theta_0 is not None else float(self.theta_states[self.initial_state])

        # Warn if user-specified theta_0 doesn't match initial state
        if self.theta_0 is not None and not math.isclose(self.theta_0, self.theta_states[self.initial_state], rel_tol=1e-6):
            warnings.warn(
                f"theta_0 ({self.theta_0}) does not match the target angle for initial state "
                f"({self.theta_states[self.initial_state]}). Simulation will start at theta_0."
            )

        # Step 2: Initialize arrays to store simulation data
        bead_positions = np.zeros(self.steps, dtype=float)  # Bead angles
        states = np.zeros(self.steps, dtype=int)  # Motor states
        target_thetas = np.zeros(self.steps, dtype=float)  # Target angles for each step

        # Step 3: Set initial values
        bead_positions[0] = theta_start
        states[0] = int(self.initial_state)
        target_thetas[0] = float(self.theta_states[self.initial_state])
        cycle_count = 0  # Counts full rotations; used for target theta updates

        # Step 4: Main simulation loop
        for i in range(1, self.steps):
            # 4a. Determine the next state probabilistically
            new_state, delta_cycle = self._sample_transition(states[i - 1], rand_gen)  # Randomly Pick next state based on transition matrix
            cycle_count += delta_cycle  # Update cycle count if a full rotation occurred

            # 4b. Compute the new target angle based on the current state and completed cycles
            new_theta = self._update_theta(new_state, cycle_count)

            # 4c. Record state and target angle
            states[i] = new_state
            target_thetas[i] = new_theta

            # 4d. Bead angle update
            bead_positions[i] = self._update_angle(bead_positions[i - 1], new_theta, method_lowered, rand_gen)

        return bead_positions, states, target_thetas

    @staticmethod
    def computeGammaB(a: float = 0, r: float = 0, eta: float = 0) -> float:
        # Compute rotational friction coefficient of the bead.
        return 8 * math.pi * eta * a**3 + 6 * math.pi * eta * a * r**2

    # -------------------- Helper functions --------------------

    def _sample_transition(self, prev_state, rand_gen):
        """Sample chemical/structural transition from prev_state probabilistically.

        Args:
            prev_state: the current discrete state of the system
            rand_gen: random number generator object

        Returns:
            new_state: the next state after the transition
            delta_cycle: ±1 if a full rotation occurred, else 0
        """

        #  Grab outgoing rates
        current_rates = self.transition_matrix[prev_state, :]  # All rates from current state
        outgoing_rates = np.delete(current_rates, prev_state)  # Remove self-transition, we don't care if we can transition into the same state
        reaction_indices = np.delete(np.arange(len(current_rates)), prev_state)  # List of states we can move to

        #  Transition probability for this time-step
        total_rate = outgoing_rates.sum()  # Overall speed of leaving the current state
        p_react = 1.0 - math.exp(-total_rate * self.dt) if total_rate > 0 else 0.0  #  Probability of any state change this dt (Poisson statistics)

        new_state = prev_state
        delta_cycle = 0

        #  Decide if transition occurs
        if rand_gen.random() < p_react:  # If true, meaning a state change will occur
            probs = outgoing_rates / total_rate  # np.array: the probabilities of jumping to each state, if a jump happens
            cumulative = np.cumsum(probs)  # probs → CDF
            chosen_idx = int(np.searchsorted(cumulative, rand_gen.random(), side="right"))  # Weighted CDF pick, returns the bin it falls into
            new_state = reaction_indices[chosen_idx]
            # Count full rotations
            if prev_state == len(self.theta_states) - 1 and new_state == 0:
                delta_cycle = 1
            elif prev_state == 0 and new_state == len(self.theta_states) - 1:
                delta_cycle = -1

        return new_state, delta_cycle

    def _update_theta(self, state, cycle_count):
        """Compute the target angle for the bead given state and rotation count.

        Each state has a base angle (theta_states[state]). For every complete
        4-state rotation, add 2π/3 rad (120 deg) to keep the angle continuous.
        """
        return float(self.theta_states[state]) + (cycle_count) * 2 * math.pi / 3

    def _update_angle(self, prev_angle, target_theta, method_lowered, rand_gen):
        # Update bead position using the chosen integration method.
        match method_lowered:
            case "heun":
                return self._heun_1d(prev_angle, target_theta, rand_gen)
            case "euler":
                return self._euler_maruyama(prev_angle, target_theta, rand_gen)
            case "probabilistic":
                return self._probabilistic(prev_angle, target_theta, rand_gen)
            case _:
                raise ValueError("Method must be 'heun', 'euler', or 'probabilistic'")

    def _verify_attributes(self):
        # Ensure all required attributes have been set before simulation.
        required_attributes = ["steps", "dt", "gammaB", "kappa", "kBT", "method", "transition_matrix", "initial_state", "theta_states"]
        missing = [attr for attr in required_attributes if getattr(self, attr) is None]
        if missing:
            raise ValueError(f"Missing attributes: {missing}")

    def _drift(self, current_angle, theta_target):
        # Deterministic drift term: tendency of bead to relax toward target angle.
        return -self.kappa * (current_angle - theta_target) / self.gammaB

    def _diffusion(self):
        # Stochastic diffusion term magnitude for bead motion.
        return math.sqrt(2 * self.kBT / self.gammaB)

    def _heun_1d(self, current_angle, theta_target, rand_gen):
        # Heun integration (predictor-corrector) for 1D bead motion.
        driftX = self._drift(current_angle, theta_target)
        diffusionX = self._diffusion()
        eta = rand_gen.gauss(0, 1)

        # Predictor step
        y_predict = current_angle + self.dt * driftX + math.sqrt(self.dt) * diffusionX * eta
        # Corrector step
        drift_predict = self._drift(y_predict, theta_target)
        return current_angle + (self.dt / 2) * (driftX + drift_predict) + math.sqrt(self.dt) * diffusionX * eta

    def _euler_maruyama(self, current_angle, theta_target, rand_gen):
        # Euler-Maruyama stochastic integration for bead motion.
        return (
            current_angle
            + self.dt * self._drift(current_angle, theta_target)
            + math.sqrt(2 * self.kBT / self.gammaB * self.dt) * rand_gen.gauss(0, 1)
        )

    def _probabilistic(self, current_angle, theta_target, rand_gen):
        # Probabilistic update using Ornstein-Uhlenbeck analytical solution.
        exp_factor = math.exp(-self.kappa / self.gammaB * self.dt)
        mean = theta_target + (current_angle - theta_target) * exp_factor
        std_dev = math.sqrt(self.kBT / self.kappa * (1 - exp_factor**2))
        return mean + std_dev * rand_gen.gauss(0, 1)
