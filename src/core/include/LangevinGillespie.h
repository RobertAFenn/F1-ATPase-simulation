#pragma once

#include <cmath>
#include <optional>
#include <utility>   // std::pair
#include <random>    // std::mt19937
#include <stdexcept> // std::runtime_error
#include <string>
#include <tuple>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for std::vector and std::tuple conversion
#include <pybind11/numpy.h>

namespace py = pybind11;
/**
 * @class LangevinGillespie
 * @brief Implements a multi-state Langevin dynamics simulation with discrete transitions.
 *
 * Implementation is split between:
 *  - LangevinGillespie.cpp (src/core/cpp/LangevinGillespie.cpp) : CPU functions (simulate, simulate_multithreaded, computeGammaB)
 *  - LangevinGillespie.cu  (src/core/cuda/LangevinGillespie.cu) : GPU functions (simulate_multithreaded_gpu)
 *
 * Pybind11 bindings can be found in binding.cpp (src/core/binding.cpp)
 *
 * The LangevinGillespie class simulates a bead subject to thermal noise and elastic forces
 * that transitions between discrete chemical states. Each state corresponds to a target angle,
 * and transitions occur probabilistically according rates held within a transition matrix.
 */
class LangevinGillespie {
public:
    // -=-=-=-=-=-=-=-=-= Simulation Parameters -=-=-=-=-=-=-=-=-=
    std::optional<unsigned int> steps; /** @brief  Number of simulation time steps */
    std::optional<double> dt;          /** @brief Time step size */
    std::optional<std::string> method; /** @brief Integration method:  "heun", "euler", or "probabilistic"*/

    // -=-=-=-=-=-=-=-=-= Physical System Parameters -=-=-=-=-=-=-=-=-=
    std::optional<double> theta_0; /** @brief Initial bead position  (radians)*/
    std::optional<double> kappa;   /** @brief Elastic constant of  the system (pN.nm/rad²)*/
    std::optional<double> kBT;     /** @brief Thermal energy (pN .nm)*/
    std::optional<double> gammaB;  /** @brief Rotational friction coefficient  of the bead (pN.nm.s)*/

    // -=-=-=-=-=-=-=-=-= Multi-state Parameters -=-=-=-=-=-=-=-=-=
    std::optional<std::vector<std::vector<double>>> transition_matrix; /** @brief Transition rate matrix  between states*/
    std::optional<unsigned int> initial_state;                         /** @brief Index of starting  state*/
    std::optional<std::vector<double>> theta_states;                   /** @brief Target angles for  each state (in radians)*/

    /**
     * @brief Parameters for LangevinGillespie simulations, only used within CUDA code
     *
     * CUDA device code cannot directly access full C++ classes because of restrictions on memory layout and dynamic features.
     * Therefore, a plain struct is required to pass simulation parameters to the GPU. For this reason, this parameter
     * must remain public. Use LangevinGillespie::toStruct() to convert a class instance to this struct.
     */
    struct LGParams {
        // -=-=-=-=-=-=-=-=-= Simulation Parameters -=-=-=-=-=-=-=-=-=
        unsigned int steps; /** @brief  Number of simulation time steps */
        float dt;           /** @brief Time step size */
        int method;         /** @brief Integration method:  1 -> "heun", 2 -> "euler", or 3 -> "probabilistic"*/

        // -=-=-=-=-=-=-=-=-= Physical System Parameters -=-=-=-=-=-=-=-=-=
        float theta_0;  /** @brief Initial bead position  (radians)*/
        float kappa;    /** @brief Elastic constant of  the system (pN.nm/rad²)*/
        float kBT;      /** @brief Thermal energy (pN .nm)*/
        float gammaB;    /** @brief Rotational friction coefficient  of the bead (pN.nm.s)*/

        // -=-=-=-=-=-=-=-=-= Multi-state Parameters -=-=-=-=-=-=-=-=-=
        unsigned int initial_state;     /** @brief Transition rate matrix  between states*/
        float theta_states[4];          /** @brief Index of starting  state*/
        float transition_matrix[4 * 4]; // Flattened 2D matrix /** @brief Target angles for  each state (in radians), flattened 2D matrix*/ 
    };


    /** @return new instance of LangevinGillespie */
    LangevinGillespie() = default;

    // -=-=-=-=-=-=-=-=-= Methods -=-=-=-=-=-=-=-=-=

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
     *  - std::vector<double> bead_positions: Bead angles over time
     *  - std::vector<int> states: Discrete states over time
     *  - std::vector<double> target_thetas: Target angles for each step
     */
    std::tuple<std::vector<double>, std::vector<int>, std::vector<double>>
        simulate(const std::optional<unsigned int>& seed = std::nullopt);


    /**
     *  @brief Run multiple LangevinGillespie simulations in parallel. nSim is distributed evenly among thread count
     *
     *  Distributes nSim simulations evenly among the number of threads specified. Each simulation will run independently,
     *  then store its results. If a seed is provided, it will be used to initialize RNG for reproducibility. Otherwise,
     *  each simulation will be randomized independently.
     *
     *  The per simulation seed is calculated using the following formula
     *  sim_seed = seed.value() + thread_start_idx + sim
     *
     *  @param nSim The total number of simulations to run
     *  @param num_threads The number of threads to use for parallel execution
     *  @param seed Optional RNG integer for local reproducibility. If not provided, simulations are fully randomized.
     *
     *  @returns A tuple containing three 2D Vectors
     *  - std::vector<std::vector<double>> bead_positions: Angles of beads for each simulation over time // TODO convert to numpy arrays for consistency?
     *  - std::vector<std::vector<int>> states: States for each simulation over time
     *  - std::vector<std::vector<double>> target_thetas: Target thetas for each simulation over time
     * */
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<int>>, std::vector<std::vector<double>>>
        simulate_multithreaded(
            unsigned int nSim,
            unsigned int num_threads,
            const std::optional<unsigned int>& seed = std::nullopt
        );

    /**
     * @brief Run multiple LangevinGillespie simulations on the GPU through CUDA
     *
     * This method runs on the GPU. Unlike 'simulate_multithreaded" (CPU), each thread on the GPU will run an instance of
     * the simulation. This allows for large parallelization by utilizing CUDA cores.
     *
     * @param nSim The total number of simulations to run
     * @param seed Optional RNG integer for local reproducibility. If not provided, simulations are fully randomized.
     *
     * @return A std::tuple containing three Numpy arrays:
     * - py::array_t<float> bead_positions: Angles of beads for each simulation over time
     * - py::array_t<int> states:  States for each simulation over time
     * - py::array_t<float> target_thetas: Target thetas for each simulation over time
     *
     * @note CUDA device code cannot access full C++ classes. To navigate this issue, a struct is passed internally.
     *       Use 'to_struct' to convert the class if needed
     */
    std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<float>>
        simulate_multithreaded_cuda(int nSim, unsigned long long seed);


    /**
     * @brief Compute rotational friction coefficient of the bead.
     * @param a Bead radius (nm)
     * @param r (nm)
     * @param eta (pN.s/nm²)
     * */
    double computeGammaB(double a, double r, double eta);

private:
    // -=-=-=-=-=-=-=-=-= GPU only parameters -=-=-=-=-=-=-=-=-=

    float* d_beads = nullptr;           /** @brief  Device buffer for bead_angles (GPU only, fallback is zero-copy mapping is unavailable)*/
    float* d_thetas = nullptr;          /** @brief  Device buffer for target_thetas (GPU only, fallback is zero-copy mapping is unavailable)*/
    int* d_states = nullptr;            /** @brief  Device buffer for states (GPU only, fallback is zero-copy mapping is unavailable)*/
    LGParams* d_params = nullptr;       /** @brief  Device copy of simulation parameters struct*/
    size_t current_allocated_size = 0;  /** @brief  Tracks current size of the device buffers, in order to avoid repeated cudaMalloc calls*/

    // -=-=-=-=-=-=-=-=-= Helper Methods -=-=-=-=-=-=-=-=-=

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

    /**
     * @brief Converts the current LangevinGillespie class parameters into a cuda compatable LGParams struct
     *
     * This function packages all class parameters into the a struct suitable for the GPU kernel.
     * Points of interest:
     * - Converts method strings to integers (heun → 0, euler → 1, probabilistic → 2)
     * - Handles optional theta_0, if null uses theta_states[initial_state]
     * - Copies up to 4 target angles into theta states, the rest are zeroed
     * - Flattens a 4x4 transition matrix into a row major float array. Missing entries are zeroed
     * - Converts all double values to float in order to reduce CUDA device usage
     *
     * @return LGParams Struct containing all public parameters
     */
    LGParams to_struct() const;
};
