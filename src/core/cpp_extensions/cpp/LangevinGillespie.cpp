#include "../include/LangevinGillespie.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for std::vector and std::tuple conversion
#include <algorithm>      // for std::min, std::max, std::clamp, std::find, etc.
#include <cctype>         // for std::tolower (if you normalize "method")
#include <iostream>       
#include <numeric>        // for std::accumulate (if you sum transition rates)
#include <numbers>        // math constants


// --------------------- Pybind11 Wrap ---------------------
namespace py = pybind11;

PYBIND11_MODULE(f1sim, m) {
    m.doc() = "Pybind11 wrap that exposing the LangevinGillespie class";

    py::class_<LangevinGillespie>(m, "LangevinGillespie")
        .def(py::init<>()) // Default constructor

        .def_readwrite("steps", &LangevinGillespie::steps)
        .def_readwrite("dt", &LangevinGillespie::dt)
        .def_readwrite("method", &LangevinGillespie::method)

        .def_readwrite("theta_0", &LangevinGillespie::theta_0)
        .def_readwrite("kappa", &LangevinGillespie::kappa)
        .def_readwrite("kBT", &LangevinGillespie::kBT)
        .def_readwrite("gammaB", &LangevinGillespie::gammaB)

        .def_readwrite("transition_matrix", &LangevinGillespie::transition_matrix)
        .def_readwrite("initial_state", &LangevinGillespie::initial_state)
        .def_readwrite("theta_states", &LangevinGillespie::theta_states)

        .def("simulate", &LangevinGillespie::simulate,
            py::arg("seed") = std::nullopt,
            "Run a Langevin simulation.\n"
            "Returns:\n"
            "    bead_positions: array of bead angles over time\n"
            "    states: array of discrete states over time\n"
            "    target_thetas: array of target angles for each step")

        .def("computeGammaB", &LangevinGillespie::computeGammaB,
            py::arg("a") = 0,
            py::arg("r") = 0,
            py::arg("eta") = 0,
            "Compute rotational friction coefficient of the bead.");
}

// --------------------- Public Methods ---------------------
std::tuple<std::vector<double>, std::vector<int>, std::vector<double>>
LangevinGillespie::simulate(const std::optional<unsigned int>& seed) {
    // Step -2: Check that all required attributes have been set
    LangevinGillespie::verify_attributes();

    // Step -1: Set up random number generation
    std::mt19937 rand_gen = LangevinGillespie::create_rng(seed);

    std::string method_lowered = LangevinGillespie::method.value();
    std::transform(method_lowered.begin(), method_lowered.end(), method_lowered.begin(),
        [](unsigned char c)
        { return std::tolower(c); });

    // Step 1: Determine starting angle (local variable, does not overwrite self.theta_0)
    double theta_start = theta_0.has_value()
        ? theta_0.value()
        : theta_states.value().at(initial_state.value());

    // Warn if user-specified theta_0 doesn't match initial state
    if (theta_0.has_value())
    {
        double target = theta_states.value().at(initial_state.value());
        double rel_tol = 1e-6;
        if (std::abs(theta_0.value() - target) > rel_tol * std::abs(target)) // Check if self.theta_0 is close to target
        {
            std::cerr << "Warning: theta_0 (" << theta_0.value()
                << ") does not match the target angle for initial state ("
                << target << "). Simulation will start at theta_0."
                << std::endl;
        }
    }

    // Step 2: Initialize Arrays to store simulation data
    std::vector<double> bead_positions(LangevinGillespie::steps.value());
    std::vector<int> states(LangevinGillespie::steps.value());
    std::vector<double> target_thetas(LangevinGillespie::steps.value());

    // Step 3: Set Initial Values
    bead_positions[0] = theta_start;
    states[0] = LangevinGillespie::initial_state.value();
    target_thetas[0] = LangevinGillespie::theta_states.value()[LangevinGillespie::initial_state.value()];
    unsigned int cycle_count = 0;

    // Step 4: Main Simulation Loop
    for (size_t i = 1; i < LangevinGillespie::steps.value(); i++) {

        // 4a. Determine the next state probabilistically
        std::pair<int, int> result = sample_transition(states[i - 1], rand_gen); // Randomly pick next state based off transition matrix
        int new_state = result.first;
        int delta_cycle = result.second;
        cycle_count += delta_cycle;

        // 4b. Compute the new target angle based on the current state and completed cycles
        double new_theta = LangevinGillespie::update_theta(new_state, cycle_count);

        // 4c. Record state and target angle
        states[i] = new_state;
        target_thetas[i] = new_theta;

        // 4d. Bead angle update
        bead_positions[i] = LangevinGillespie::update_angle(bead_positions[i - 1], new_theta, method_lowered, rand_gen);

    }

    return{ bead_positions, states, target_thetas };
}

double LangevinGillespie::computeGammaB(double a, double r, double eta) {
    return 8 * std::numbers::pi * eta * (a * a * a) + 6 * std::numbers::pi * eta * a * (r * r);
}

// --------------------- Private Helper Methods ---------------------
void LangevinGillespie::verify_attributes() const {
    std::vector<std::string> missing;

    if (!steps.has_value())
        missing.push_back("steps");
    if (!dt.has_value())
        missing.push_back("dt");
    if (!method.has_value())
        missing.push_back("method");
    if (!kappa.has_value())
        missing.push_back("kappa");
    if (!kBT.has_value())
        missing.push_back("kBT");
    if (!gammaB.has_value())
        missing.push_back("gammaB");
    if (!transition_matrix.has_value())
        missing.push_back("transition_matrix");
    if (!initial_state.has_value())
        missing.push_back("initial_state");
    if (!theta_states.has_value())
        missing.push_back("theta_states");

    if (!missing.empty()) {
        std::string msg = "Missing attributes: ";
        for (const auto& attr : missing)
            msg += "[" + attr + "]";

        throw std::runtime_error(msg);
    }
}

std::mt19937 LangevinGillespie::create_rng(const std::optional<unsigned int>& seed) {
    {
        if (seed.has_value())
            return std::mt19937(seed.value()); // Seeded RNG

        std::random_device rd;
        return std::mt19937(rd()); // Non-deterministic seed

    }
}

std::pair<int, int>
LangevinGillespie::sample_transition(int prev_state, std::mt19937& local_rng) const {
    const std::vector<double>& current_rates = transition_matrix.value()[prev_state];
    std::vector<double> outgoing_rates;
    std::vector<int> reaction_indices;

    // Removes self transition in outgoing_rates, then we are given the states we can go to 
    for (size_t i = 0; i < current_rates.size(); ++i) {
        if (static_cast<int>(i) != prev_state) {
            outgoing_rates.push_back(current_rates[i]);
            reaction_indices.push_back(i);
        }
    }

    double total_rate = std::accumulate(outgoing_rates.begin(), outgoing_rates.end(), 0.0);
    double p_react = (total_rate > 0) ? 1.0 - std::exp(-total_rate * LangevinGillespie::dt.value()) : 0.0;

    int new_state = prev_state;
    int delta_cycle = 0;

    // Decide if a transition occurs
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (dist(local_rng) < p_react) { // If true, meaning a state change occurs

        // Get probabilities and CDF
        std::vector<double> probs(outgoing_rates.size());
        for (size_t i = 0; i < outgoing_rates.size(); ++i)
            probs[i] = outgoing_rates[i] / total_rate; // probabilities of jumping to each state

        std::vector<double> cumulative(probs.size());
        std::partial_sum(probs.begin(), probs.end(), cumulative.begin()); // probs → CDF

        // Select Next State
        double r = dist(local_rng);
        auto it = std::upper_bound(cumulative.begin(), cumulative.end(), r);
        size_t idx = std::distance(cumulative.begin(), it);
        if (idx >= reaction_indices.size()) idx = reaction_indices.size() - 1; // clamp
        new_state = reaction_indices[idx];

        // Count full rotations
        if (prev_state == static_cast<int>(LangevinGillespie::theta_states.value().size()) - 1 && new_state == 0)
            delta_cycle = 1;
        else if (prev_state == 0 && new_state == static_cast<int>(LangevinGillespie::theta_states.value().size()) - 1)
            delta_cycle = -1;


    }
    return { new_state, delta_cycle };

}

double LangevinGillespie::update_theta(int state, int cycle_count) const {
    return LangevinGillespie::theta_states->at(state) + (cycle_count) * 2 * std::numbers::pi / 3;
}

double LangevinGillespie::update_angle(double prev_angle, double target_theta, const std::string& method_lowered, std::mt19937& local_rng) const {
    if (method_lowered == "heun")
        return LangevinGillespie::heun_1d(prev_angle, target_theta, local_rng);

    else if (method_lowered == "euler")
        return LangevinGillespie::euler_maruyama(prev_angle, target_theta, local_rng);

    else if (method_lowered == "probabilistic")
        return LangevinGillespie::probabilistic(prev_angle, target_theta, local_rng);

    else {
        throw std::runtime_error("Method must be 'heun', 'euler', or 'probabilistic'");

    }
}

double LangevinGillespie::drift(double current_angle, double target_theta) const {
    return -LangevinGillespie::kappa.value() * (current_angle - target_theta) / LangevinGillespie::gammaB.value();
}

double LangevinGillespie::diffusion() const {
    return std::sqrt(2 * LangevinGillespie::kBT.value() / LangevinGillespie::gammaB.value());
}

double LangevinGillespie::heun_1d(double current_angle, double theta_target, std::mt19937& local_rng) const {
    double drift_term = drift(current_angle, theta_target);
    double diffusion_term = diffusion();
    std::normal_distribution<double> dist(0.0, 1.0);
    double eta = dist(local_rng); // random number ~ N(0,1)

    // Predictor step
    double y_predict = current_angle + LangevinGillespie::dt.value() * drift_term + std::sqrt(LangevinGillespie::dt.value()) * diffusion_term * eta;

    // Corrector step
    double drift_predict = drift(y_predict, theta_target);
    return current_angle + (LangevinGillespie::dt.value() / 2) * (drift_term + drift_predict) + std::sqrt(LangevinGillespie::dt.value()) * diffusion_term * eta;
}

double LangevinGillespie::euler_maruyama(double current_angle, double theta_target, std::mt19937& local_rng) const {
    double drift_term = drift(current_angle, theta_target);
    double diffusion_term = diffusion() * std::sqrt(LangevinGillespie::dt.value());
    std::normal_distribution<double> dist(0.0, 1.0);
    double eta = dist(local_rng); // random number ~ N(0,1)
    return current_angle + LangevinGillespie::dt.value() * drift_term + diffusion_term * eta;
}

double LangevinGillespie::probabilistic(double current_angle, double target_theta, std::mt19937& local_rng) const {
    double exp_factor = std::exp(-LangevinGillespie::kappa.value() / LangevinGillespie::gammaB.value() * LangevinGillespie::dt.value());
    double mean = target_theta + (current_angle - target_theta) * exp_factor;
    double std_dev = std::sqrt(LangevinGillespie::kBT.value() / LangevinGillespie::kappa.value() * (1 - (exp_factor * exp_factor)));
    std::normal_distribution<double> dist(0.0, 1.0);
    double eta = dist(local_rng);
    return mean + std_dev * eta;
}