import numpy as np
import math


"""
NOTE: Since this code will most likely be viewed by people who I assume mainly use matlab, and may not
      be familiar with numpy, I will make comments to help better understand. If you are still confused
      I highly recommend checking out the documentation for numpy.

The `compute_angular_profiles` function calculates angular profiles and jump statistics based on input
data. It has three main goals:
1) Track how the system's angle changes over time.
2) Compute the average angular jump (how the angle changes between steps, i.e., angular velocity) 
   at differing reference angles.
3) Build a distribution of the angular jumps, so we know both the average and the spread of 
   possible jumps.

--- Parameters ---
dt : Time step between consecutive data points (ms)
pos_store : Position data in radians, shape (time_steps, num_simulations)
param state_store : State information corresponding to positions, shape matches pos_store

--- Returns ---
Four arrays
- fine_profile_matrix: [reference angle, mean jump] for fine-grained data
- fine_jump_pdf_matrix: PDFs of jumps for fine-grained data
- coarse_profile_matrix: [reference angle, mean jump] for coarse-grained data
- coarse_jump_pdf_matrix: PDFs of jumps for coarse-grained data
"""


def compute_angular_profiles(dt, pos_store, state_store):
    # Prepare fine and coarse resolution datasets
    fine_data, coarse_data = prep_fine_and_coarse(pos_store, state_store, dt)
    fine_positions_deg, fine_velocity_deg_per_ms, fine_states = fine_data
    coarse_positions_deg, coarse_velocity_deg_per_ms, coarse_states = coarse_data

    # Reference angles
    reference_angle_step = 3
    reference_angles = np.arange(-30, 121, reference_angle_step)
    angle_tolerance = 3  # ±3° around each reference angle, we dont need an exact match

    # Histogram variables: used to see how frequently different angular jumps occur
    jump_bin_width = 200  # Each bin covers 200 deg/ms
    jump_bins = np.linspace(-1e5, 1e5, num=int(2e5 / jump_bin_width) + 1)  # Bin edges to track jump frequencies

    # Compute fine-grained profile and PDFs (Probability Density Function)
    fine_profile_matrix, fine_jump_pdf_matrix = compute_profile(
        fine_positions_deg, fine_velocity_deg_per_ms, fine_states, reference_angles, angle_tolerance, jump_bins
    )

    # Compute coarse-grained profile and PDFs (Probability Density Function)
    coarse_profile_matrix, coarse_jump_pdf_matrix = compute_profile(
        coarse_positions_deg, coarse_velocity_deg_per_ms, coarse_states, reference_angles, angle_tolerance, jump_bins
    )

    return fine_profile_matrix, fine_jump_pdf_matrix, coarse_profile_matrix, coarse_jump_pdf_matrix


# Helper Functions
def prep_fine_and_coarse(pos_store, state_store, dt, cstep=10):
    """
    Prepare the position and state data at two resolutions:
    1. Fine resolution: detailed, every time step ("HD view")
    2. Coarse resolution: smoother, every cstep-th time step (removes high-frequency noise)
    """

    # Convert positions from radians to degrees
    positions_deg = pos_store * 180 / math.pi

    # Fine resolution // "HD view"
    fine_positions_deg = positions_deg[:-1, :]
    fine_velocity_deg_per_ms = np.diff(positions_deg, axis=0) / (dt * 1e3)  #  Read this as velocity = (deg2-deg1) / ms
    fine_states = state_store[:-1, :]

    # Coarse resolution //  "general view"
    coarse_positions_deg_full = positions_deg[::cstep, :]  # [::cstep, :] meaning every cstep-th step // reduces data points
    coarse_states_full = state_store[::cstep, :]

    coarse_positions_deg = coarse_positions_deg_full[:-1, :]
    coarse_velocity_deg_per_ms = np.diff(coarse_positions_deg_full, axis=0) / (cstep * dt * 1e3)
    coarse_states = coarse_states_full[:-1, :]

    # Return fine and course data as tuples
    return (fine_positions_deg, fine_velocity_deg_per_ms, fine_states), (coarse_positions_deg, coarse_velocity_deg_per_ms, coarse_states)


def compute_profile(positions_deg, velocities_deg_per_ms, states, reference_angles, angle_tolerance, jump_bins):
    # Computes the mean jump and jump PDFs for a given dataset at specified reference angles.

    n_bins = len(reference_angles)  # Number of reference angles we analyze

    profile_matrix = np.full((n_bins, 2), np.nan)  # [ref angle, mean jump]
    jump_pdf_matrix = np.full((len(jump_bins) - 1, n_bins), np.nan)  # PDF of jumps per reference angle

    for i, ref_angle in enumerate(reference_angles):
        profile_matrix[i, 0] = ref_angle
        profile_matrix[i, 1], jump_pdf_matrix[:, i] = compute_jump_statistics(
            positions_deg, velocities_deg_per_ms, states, ref_angle, angle_tolerance, jump_bins
        )

    return profile_matrix, jump_pdf_matrix


def compute_jump_statistics(positions_deg, velocities_deg_per_ms, states, reference_angle, angle_tolerance, jump_bins):
    """
    Computes the mean jump and jump probability distribution for a given reference angle.

    The pdf calculated is defined as the following
    -  A probability density function (PDF) describes how likely it
       is to observe some outcome resulting from a data-generating process

    Here the PDF tells us how likely angular jumps (change in angle per ms) at each reference angle

    --- Parameters ---
    positions_deg : Angular positions in degrees
    velocities_deg_per_ms : Angular velocity (deg/ms)
    states : State information aligned with positions
    reference_angle : The angle to evaluate (degrees)
    angle_tolerance : Window around the reference angle (± degrees)
    jump_bins : Bin edges for histogram

    --- Returns ---
    mean jump (deg), PDF of jumps
    """

    # Select data within ± angle_tolerance of the reference angle
    matching_indices = np.where(np.abs(positions_deg - reference_angle) <= angle_tolerance)

    if len(matching_indices[0]) == 0:
        return np.nan, np.zeros(len(jump_bins) - 1)  # No points found, return nan for the mean, and a zero PDF

    # Extract Jump and State
    matched_jumps = velocities_deg_per_ms[matching_indices]
    matched_states = states[matching_indices]

    # Compute Probabilities for each state
    unique_states = np.unique(matched_states)
    state_counts = np.array([np.sum(matched_states == s) for s in unique_states])
    state_probabilities = state_counts / np.sum(state_counts)

    # Initialize combined PDF: stores PDF of angular jumps
    jump_pdf = np.zeros((len(jump_bins) - 1,))

    # For each unique state, compute a histogram of jumps and weight it by the state's probability
    bin_width = jump_bins[1] - jump_bins[0]  # Needed for pdf normalization
    for state, prob in zip(unique_states, state_probabilities):
        state_jumps = matched_jumps[matched_states == state]
        counts_hist, _ = np.histogram(state_jumps, bins=jump_bins)

        counts_pdf = counts_hist / (counts_hist.sum() * bin_width) if counts_hist.sum() != 0 else np.zeros_like(counts_hist)
        jump_pdf += counts_pdf * prob  # Weigh the states PDF by the probability, and add it to the total PDF

    # Compute mean jump from PDF
    bin_centers = (jump_bins[:-1] + jump_bins[1:]) / 2
    mean_jump = np.sum(jump_pdf * bin_centers * bin_width) 

    return mean_jump, jump_pdf
