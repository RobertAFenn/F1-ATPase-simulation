#pragma once

#include <cmath>
#include <optional>
#include <utility>   // std::pair
#include <random>    // std::mt19937
#include <stdexcept> // std::runtime_error
#include <string>
#include <tuple>
#include <vector>


/**
 * @class LangevinGillespie
 * @brief Implements a multi-state Langevin dynamics simulation with discrete transitions.
 *
 * The LangevinGillespie class simulates a bead subject to thermal noise and elastic forces
 * that transitions between discrete chemical states. Each state corresponds to a target angle,
 * and transitions occur probabilistically according rates held within a transition matrix.
 */
class LangevinGillespie {
public:
    // Simulation Parameters
    std::optional<unsigned int> steps; /** @brief  Number of simulation time steps */
    std::optional<double> dt;          /** @brief Time step size */
    std::optional<std::string> method; /** @brief Integration method:  "heun", "euler", or "probabilistic"*/

    // Physical System Parameters
    std::optional<double> theta_0; /** @brief Initial bead position  (radians)*/
    std::optional<double> kappa;   /** @brief Elastic constant of  the system (pN.nm/rad²)*/
    std::optional<double> kBT;     /** @brief Thermal energy (pN .nm)*/
    std::optional<double> gammaB;  /** @brief Rotational friction coefficient  of the bead (pN.nm.s)*/

    // Multi-state parameters
    std::optional<std::vector<std::vector<double>>> transition_matrix;/** @brief Transition rate matrix  between states*/
    std::optional<unsigned int> initial_state;                        /** @brief Index of starting  state*/
    std::optional<std::vector<double>> theta_states;                  /** @brief Target angles for  each state (in radians)*/

    /** @return new instance of LangevinGillespie */
    LangevinGillespie() = default;

    // --------------------- Methods ---------------------

    /**
     * @brief Run a Langevin simulation.
     *
     * Simulates a bead according to a stochastic integration method
     * State transitions occur probabilistically, modifying the target angle at each time step
     * All class parameters must be set before calling this function, except optionally theta_0
     * If theta_0 is not defined, theta_0 will be set to self.theta_states[self.theta_0]
     *
     * @param seed Optional RNG integer for local reproducibility
     * @returns A tuple containing:
     *  - bead_positions: array of bead angles over time
     *  -  states: array of discrete states over time
     *  - target_thetas: array of target angles for each step
     */
    std::tuple<std::vector<double>, std::vector<int>, std::vector<double>>
        simulate(const std::optional<unsigned int>& seed = std::nullopt);

    /**
     * @brief Compute rotational friction coefficient of the bead.
     * @param a Bead radius (nm)
     * @param r (nm)
     * @param eta (pN.s/nm²)
     * */
    double computeGammaB(double a, double r, double eta);

private:
    // --------------------- Helper Methods ---------------------

    /**
     * @brief Ensures that all required attributes are properly initialized before simulation.
     *
     * This function verifies that all necessary parameters (e.g., physical constants,
     * simulation parameters, and configuration values) have been assigned valid values.
     * It should be called before the simulation loop begins.
     *
     * @throws std::runtime_error If any required attribute is missing or invalid.
     */
    void verify_attributes() const;

    /**
     * @brief Creates and initializes a local random number generator.
     *
     * This function returns a Mersenne Twister pseudo-random number generator (PRNG),
     * optionally seeded with a user-specified value for reproducibility.
     *
     * @param seed Optional seed value (unsigned integer). If not provided, a
     *             non-deterministic seed is used.
     * @return std::mt19937 A fully initialized PRNG instance.
     */
    std::mt19937 create_rng(const std::optional<unsigned int>& seed);

    /**
     * @brief Computes the next transition for the system's current state.
     *
     * Given the previous state and a random number generator, this function samples
     * the next state and rotation index according to the transition probabilities.
     * Each state corresponds to a base angle (from @c theta_states[state]). For every
     * complete 4-state rotation, 2π/3 radians (120°) are added to maintain continuous
     * angular progression.
     *
     * @param prev_state The index of the current state.
     * @param local_rng Reference to an initialized local random number generator.
     * @return std::pair<int, int> The next state index and updated rotation count.
     */
    std::pair<int, int>
        sample_transition(int prev_state, std::mt19937& local_rng) const;

    /**
     * @brief Computes the target angle given the current state and completed cycles
     *
     * Compute the target angle for the bead given state and rotation count.
     * Each state has a base angle (theta_states[state]). For every complete
     * 4-state rotation, add 2π/3 rad (120 deg) to keep the angle continuous.
     *
     * @param prev_state The index of the previous state
     * @param local_rng Reference to an initialized local random number generator
     * @return Updated target angle (radians)
     */
    double update_theta(int state, int cycle_count) const;

    /**
     * @brief Update bead angle using the chosen integration method.
     *
     * Selects between Heun, Euler-Maruyama, or probabilistic integration methods
     * to update the bead angle stochastically.
     *
     * @param prev_angle Previous bead angle (radians).
     * @param target_theta Equilibrium angle (radians).
     * @param method_lowered Integration method in lowercase ("heun", "euler", or "probabilistic").
     * @param local_rng Reference to the random number generator.
     * @throws std::invalid_argument Thrown if method_lowered is not one of the valid methods.
     * @return Updated bead angle (radians).
     */
    double update_angle(double prev_angle, double target_theta, const std::string& method_lowered, std::mt19937& local_rng) const;


    /**
     * @brief Deterministic drift term: tendency of bead to relax toward target angle.
     *
     * @param current_angle (radians)
     * @param target_theta equilibrium angle (radians)
     * @return Deterministic drift velocity (radians/second)
     */
    double drift(double current_angle, double target_theta) const;

    /**
     * @brief Computes the diffusion magnitude based off thermal noise.
     * @return Diffusion coefficient (radians/√second)
     */
    double diffusion() const;

    /**
     * @brief Heun's integration method for stochastic differential equations.
     *
     * Predictor corrector method, offers better performance than Euler-Maruyama and Probabilistic
     *
     * @param current_angle (radians)
     * @param theta_target (radians)
     * @param local_rng Reference to initialized local random generator
     * @return Updated bead angle
     */
    double heun_1d(double current_angle, double theta_target, std::mt19937& local_rng) const;

    /**
     * @brief Euler-Maruyama integration for stochastic differential equations.
     *
     * First order approximation of the Langevin equation
     *
     * @param current_angle (radians)
     * @param theta_target (radians)
     * @param local_rng Reference to initialized local random generator
     * @return Updated bead angle
     */
    double euler_maruyama(double current_angle, double theta_target, std::mt19937& local_rng) const;

    /**
     * @brief Probabilistic integration method for stochastic differential equations.
     *
     * Samples from the Gaussian distributed solution of the Langevin equation.
     *
     * @param current_angle (radians)
     * @param theta_target (radians)
     * @param local_rng Reference to initialized local random generator
     * @return Updated bead angle
     */
    double probabilistic(double current_angle, double target_theta, std::mt19937& local_rng) const;
};
