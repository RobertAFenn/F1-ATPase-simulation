import numpy as np  # type: ignore
import math

"""
Transition Rate Matrix for F1-ATPase Motor
Structure:
- Rows: Current state (0-3) [3°, 36°, 72°, 116° positions]
- Columns: Next state  
- Elements: Transition rate (transitions/second)

Key Behaviors:
1) Fast progression to State 2 (0→1: 63k/s, 1→2: 7k/s)
2) Bottleneck at States 2-3 (2→3: 2.8k/s with 350/s reverse)
3) Slow cycle completion (3→0: 800/s hydrolysis)

Rate Interpretation:
- Each number in the matrix is how many times per second the system would jump from one state to another if unimpeded.
- Higher numbers → faster transitions; lower numbers → slower transitions.
- Simulation scales these rates (which are in per second) by the time-step dt to get the probability of a jump in that step.

Final array output
array([[    0.  , 63370.5 ,     0.  ,    14.19],
       [    0.78,     0.  ,  7203.51,     0.  ],
       [    0.  ,   350.42,     0.  ,  2797.42],
       [  800.  ,     0.  ,   304.99,     0.  ]])

Forward rates (per dt)
state 0 → 1 63,370.5 * 1e-6 = 0.0633705
state 1 → 2 7,203.51 * 1e-6 = 0.00720351 
state 2 → 3 2797.42  * 1e-6 = 0.00279742 
state 3 → 0 800      * 1e-6 = 0.0008 

state 0 → 3 14.19   * 1e-6  = 0.00001419 
state 1 → 0 0.78    * 1e-6  = 0.00000078 
state 2 → 1 350.42   * 1e-6 = 0.00035042 
state 3 → 2 304.99   * 1e-6 = 0.00030499
"""


# Generally comments after a `\` refer to the legacy codebase, to maintainers this can be safely ignored
def compute_transition_matrix(LG):
    """
    Compute transition rate matrix (transition_matrix property) for F1-ATPase motor.

    Args:
        LG: LangevinGillespie object with kappa, kBT, and theta_states

    Returns:
        transition_matrix (np.array): 4x4 transition rate matrix
    """
    bio_params = create_biochemical_parameters()

    transition_rates = compute_transition_rates(LG, bio_params)

    return build_transition_matrix(transition_rates)


# -------------------- Helper functions --------------------
def create_biochemical_parameters():
    """
    Create and return biochemical parameters for F1-ATPase motor.
    The majority of the comments here relate to the old port names
    Returns:
        Dictionary of biochemical parameters
    """
    return {
        # Base rate constants
        "pi_release_rate": 5 * 1.8e4,  # 90,000 | k01
        "atp_binding_rate": 4 * 9.1e3,  # 36,400 | k02
        "adp_release_rate": 1 / (14e-6),  # ~71,429 | k21
        "hydrolysis_rate": 800,  # k03
        # Sensitivity parameters
        "pi_release_sensitivity": 6.7,  # a1
        "atp_binding_sensitivity": 0.045 * 180 / math.pi,  # ~2.578 | a2
        "adp_release_sensitivity": 0.045 * 180 / math.pi,  # ~2.578 | a21
        "hydrolysis_sensitivity": 0,  # a3
        # Concentrations
        "pi_concentration": 0.01,  # Pi
        "adp_concentration": 0.01,  # ADP
        # Bronsted slopes
        "pi_release_bronsted": 0.7,  # alpha1
        "atp_binding_bronsted": 0.5,  # alpha2
        "adp_release_bronsted": 0.5,  # alpha21
        "hydrolysis_bronsted": 0,  #  alpha3
        # Scaling factors | Reverse meaning these are applied to the backward transition rates
        "pi_reverse_rate_divisor": 1300,  # Held inside k_01
        "atp_reverse_rate_divisor": 1500,  # Held inside k_02
        "adp_reverse_rate_divisor": 1600,  # Held inside  k_21
        "hydrolysis_reverse_rate_divisor": 1600,  # Held inside k_03
    }


def compute_transition_rates(LG, bio_params):
    """
    Compute all transition rates between motor states.

    Args:
        LG: Simulation object with angles
        bio_params: Biochemical parameters dictionary

    Returns:
        Dictionary of transition rates
    """

    # Compute reverse rate constants | a_1, a_2, a_21, a_3 (lines [73, 76, 80, 83] from legacy codebase)
    reverse_constants = {
        "pi_release": compute_reverse_rate_constant(LG, bio_params["pi_release_bronsted"], LG.theta_states[0], LG.theta_states[1]),
        "atp_binding": compute_reverse_rate_constant(LG, bio_params["atp_binding_bronsted"], LG.theta_states[1], LG.theta_states[2]),
        "adp_release": compute_reverse_rate_constant(LG, bio_params["adp_release_bronsted"], LG.theta_states[2], LG.theta_states[3]),
        "hydrolysis": LG.kappa / LG.kBT * (1 - bio_params["hydrolysis_bronsted"]) * (LG.theta_states[0] + 2 * math.pi / 3 - LG.theta_states[3]),
    }

    # Compute base reverse rates | k_01, k_02, k_21, k_03 (lines [72, 75, 79, 82] from legacy codebase)
    base_reverse_rates = {
        "pi_release": (bio_params["pi_concentration"] * bio_params["pi_release_rate"] / bio_params["pi_reverse_rate_divisor"]),
        "atp_binding": (bio_params["atp_binding_rate"] / bio_params["atp_reverse_rate_divisor"]),
        "adp_release": (bio_params["adp_concentration"] * bio_params["adp_release_rate"] / bio_params["adp_reverse_rate_divisor"]),
        "hydrolysis": (bio_params["hydrolysis_rate"] / bio_params["hydrolysis_reverse_rate_divisor"]),
    }

    # Compute forward transition rates | par.k1 ⟶ k4 (lines [98 ⟶ 101] from legacy codebase)
    forward_rates = {
        "state_0_to_1": compute_base_transition_rate(bio_params["pi_release_rate"], bio_params["pi_release_sensitivity"], LG.theta_states[0]),
        "state_1_to_2": compute_base_transition_rate(bio_params["atp_binding_rate"], bio_params["atp_binding_sensitivity"], LG.theta_states[1]),
        "state_2_to_3": compute_base_transition_rate(bio_params["adp_release_rate"], bio_params["adp_release_sensitivity"], LG.theta_states[2]),
        "state_3_to_0": compute_base_transition_rate(bio_params["hydrolysis_rate"], bio_params["hydrolysis_sensitivity"], LG.theta_states[3]),
    }

    # Compute reverse transition rates | par.k_1 ⟶ k_4 (lines [103 ⟶ 106] from legacy codebase)
    reverse_rates = {
        "state_1_to_0": compute_base_transition_rate(base_reverse_rates["pi_release"], -reverse_constants["pi_release"], LG.theta_states[0]),
        "state_2_to_1": compute_base_transition_rate(base_reverse_rates["atp_binding"], -reverse_constants["atp_binding"], LG.theta_states[1]),
        "state_3_to_2": compute_base_transition_rate(base_reverse_rates["adp_release"], -reverse_constants["adp_release"], LG.theta_states[2]),
        "state_0_to_3": compute_base_transition_rate(base_reverse_rates["hydrolysis"], -reverse_constants["hydrolysis"], LG.theta_states[3]),
    }

    return {**forward_rates, **reverse_rates}  # Return 1 diction, with the keys from Forward and Reverse rates


def compute_reverse_rate_constant(LG, alpha, angle_from, angle_to):
    """
    Compute reverse rate constant using Arrhenius-like formula.

    Args:
        LG: Simulation object with physical parameters
        alpha: Bronsted slope parameter
        angle_from: Starting angle in radians
        angle_to: Target angle in radians

    Returns:
        Reverse rate constant
    """
    return LG.kappa / LG.kBT * (1 - alpha) * (angle_to - angle_from)


def compute_base_transition_rate(base_rate, sensitivity, angle):
    """
    Compute base transition rate using exponential dependence on angle.

    Args:
        base_rate: Intrinsic rate constant
        sensitivity: Angle sensitivity parameter
        angle: Current angle in radians

    Returns:
        Transition rate
    """
    return base_rate * math.exp(sensitivity * (-angle))


def build_transition_matrix(transition_rates):
    """
    Build 4x4 transition rate matrix from computed rates.

    Args:
        transition_rates: Dictionary of all transition rates

    Returns:
        4x4 transition rate matrix
    """
    return np.array(
        [
            # State 0 → (From 3° dwell)
            [0.0, transition_rates["state_0_to_1"], 0.0, transition_rates["state_0_to_3"]],  # [Forward: (Stage 0 -> 1)] [Backward: (Stage 0 -> 3)]
            # State 1 → (From 36° dwell)
            [transition_rates["state_1_to_0"], 0.0, transition_rates["state_1_to_2"], 0.0],  # [Forward: (Stage 1 -> 2)] [Backward: (Stage 1 -> 0)]
            # State 2 → (From 72° dwell)
            [0.0, transition_rates["state_2_to_1"], 0.0, transition_rates["state_2_to_3"]],  # [Forward: (Stage 2 -> 3)] [Backward: (Stage 2 -> 1)]
            # State 3 → (From 116° dwell)
            [transition_rates["state_3_to_0"], 0.0, transition_rates["state_3_to_2"], 0.0],  # [Forward: (Stage 3 -> 0)] [Backward: (Stage 3 -> 2)]
        ]
    )
