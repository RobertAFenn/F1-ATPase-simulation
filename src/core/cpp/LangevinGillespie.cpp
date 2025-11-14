// src/core/cpp/LangevinGillespie.cpp
#include "../include/LangevinGillespie.h"

// -=-=-=-=-=-=-=-=-= STD lib -=-=-=-=-=-=-=-=-=
#include <algorithm>    // std::min, std::max, std::transform
#include <cctype>       // std::tolower
#include <iostream>     // std::cerr
#include <numeric>      // std::accumulate
#include <numbers>      // std::numbers::pi
#include <random>       // std::mt19937, std::normal_distribution, std::uniform_real_distribution
#include <thread>       // std::thread
#include <sstream>      // std::ostringstream
#include <cmath>        // std::sqrt, std::abs, std::exp

// TODO TEST IMPORTS 【=◈︿◈=】 DELETE LATER 
#include <chrono>

// -=-=-=-=-=-=-=-=-= Thread-local Variables -=-=-=-=-=-=-=-=-= 
static thread_local std::normal_distribution<double> normal_dist(0.0, 1.0);
static thread_local std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

// -=-=-=-=-=-=-=-=-= Public Methods -=-=-=-=-=-=-=-=-=
std::tuple<std::vector<double>, std::vector<int>, std::vector<double>>
LangevinGillespie::simulate(const std::optional<unsigned int>& seed) {
    // Step -3: Verify attributes
    verify_attributes();

    // Step -2: Setup RNG
    std::mt19937 rand_gen = create_rng(seed);

    // Step -1: Method cleaning (cache once)
    std::string method_lowered = method.value();
    std::transform(method_lowered.begin(), method_lowered.end(), method_lowered.begin(),
        [](unsigned char c) { return std::tolower(c); });

    // Cache frequently-used optionals locally
    const auto steps_local = static_cast<size_t>(steps.value());
    const auto init_state_local = initial_state.value();
    const auto& theta_states_local = theta_states.value();

    // Step 0: Starting angle
    double theta_start = theta_0.has_value() ? theta_0.value() : theta_states_local[init_state_local];

    if (theta_0.has_value()) {
        double target = theta_states_local[init_state_local];
        double rel_tol = 1e-6;
        if (std::abs(theta_0.value() - target) > rel_tol * std::abs(target)) {
            std::cerr << "Warning: theta_0 (" << theta_0.value()
                << ") does not match the target angle for initial state ("
                << target << "). Simulation will start at theta_0.\n";
        }
    }

    // Step 1: Allocate vectors
    std::vector<double> bead_positions(steps_local);
    std::vector<int> states(steps_local);
    std::vector<double> target_thetas(steps_local);

    // Step 2: Initialize first values
    bead_positions[0] = theta_start;
    states[0] = init_state_local;
    target_thetas[0] = theta_states_local[init_state_local];
    int cycle_count = 0;

    // Step 3: Main loop
    for (size_t i = 1; i < steps_local; ++i) {
        auto [new_state, delta_cycle] = sample_transition(states[i - 1], rand_gen);
        cycle_count += delta_cycle;
        double new_theta = update_theta(new_state, cycle_count);

        states[i] = new_state;
        target_thetas[i] = new_theta;
        bead_positions[i] = update_angle(bead_positions[i - 1], new_theta, method_lowered, rand_gen);
    }

    return { std::move(bead_positions), std::move(states), std::move(target_thetas) };
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<int>>, std::vector<std::vector<double>>>
LangevinGillespie::simulate_multithreaded(
    unsigned int nSim,
    unsigned int num_threads,
    const std::optional<unsigned int>& seed
) {
    py::gil_scoped_release release;

    // Pre-allocation
    std::vector<std::vector<double>> combined_bead_positions(nSim);
    std::vector<std::vector<int>> combined_states(nSim);
    std::vector<std::vector<double>> combined_target_thetas(nSim);

    // Divide work
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    unsigned int work_per_thread = nSim / num_threads;
    unsigned int remainder = nSim % num_threads;
    unsigned int start_idx = 0;

    for (unsigned int tid = 0; tid < num_threads; ++tid) {
        unsigned int sims_for_this_thread = work_per_thread + (tid < remainder ? 1 : 0);
        unsigned int thread_start_idx = start_idx;
        start_idx += sims_for_this_thread;

        threads.emplace_back([this, sims_for_this_thread, thread_start_idx, seed,
            &combined_bead_positions, &combined_states, &combined_target_thetas]() {
                for (unsigned int sim = 0; sim < sims_for_this_thread; ++sim) {
                    std::optional<unsigned int> sim_seed;
                    if (seed.has_value()) sim_seed = seed.value() + thread_start_idx + sim;

                    auto [local_beads, local_states, local_thetas] = simulate(sim_seed);

                    unsigned int sim_idx = thread_start_idx + sim;
                    combined_bead_positions[sim_idx] = std::move(local_beads);
                    combined_states[sim_idx] = std::move(local_states);
                    combined_target_thetas[sim_idx] = std::move(local_thetas);
                }
            });
    }

    for (auto& t : threads) t.join();
    return { std::move(combined_bead_positions), std::move(combined_states), std::move(combined_target_thetas) };
}

double LangevinGillespie::computeGammaB(double a, double r, double eta) {
    return 8 * std::numbers::pi * eta * (a * a * a) + 6 * std::numbers::pi * eta * a * (r * r);
}

// -=-=-=-=-=-=-=-=-= Private Helper Methods -=-=-=-=-=-=-=-=-=
void LangevinGillespie::verify_attributes() const {
    const std::pair<bool, const char*> required_parameters[] = {
        { steps.has_value(),            "steps" },
        { dt.has_value(),               "dt" },
        { method.has_value(),           "method" },
        { kappa.has_value(),            "kappa" },
        { kBT.has_value(),              "kBT" },
        { gammaB.has_value(),           "gammaB" },
        { transition_matrix.has_value(), "transition_matrix" },
        { initial_state.has_value(),    "initial_state" },
        { theta_states.has_value(),     "theta_states" }
    };

    std::ostringstream oss;
    bool missing_any = false;
    for (const auto& param : required_parameters) {
        if (!param.first) {
            if (!missing_any) { oss << "Missing attributes: "; missing_any = true; }
            oss << "[" << param.second << "]";
        }
    }
    if (missing_any) throw std::runtime_error(oss.str());
}

std::mt19937 LangevinGillespie::create_rng(const std::optional<unsigned int>& seed) {
    if (seed.has_value()) return std::mt19937(seed.value());
    std::random_device rd;
    return std::mt19937(rd());
}

std::pair<int, int>
LangevinGillespie::sample_transition(int prev_state, std::mt19937& local_rng) const {
    // Cache commonly used values
    const auto& t_matrix = transition_matrix.value();
    const auto dt_local = dt.value();
    const size_t nStates = t_matrix[prev_state].size();


    std::vector<double> outgoing_rates; // All rates leaving prev_states to other states
    std::vector<int> reaction_indices; // Holds corresponding indices to target states
    outgoing_rates.reserve(nStates - 1);
    reaction_indices.reserve(nStates - 1);

    // Removes self transition in outgoing_rates, then get the states the simulation can go to
    for (size_t i = 0; i < nStates; i++) {
        if (static_cast<int>(i) != prev_state) {
            outgoing_rates.push_back(t_matrix[prev_state][i]);
            reaction_indices.push_back(static_cast<int>(i));
        }
    }

    int new_state = prev_state;
    int delta_cycle = 0;
    double total_rate = std::accumulate(outgoing_rates.begin(), outgoing_rates.end(), 0.0); // Sum of all transition rates from our current state
    double p_react = (total_rate > 0.0) ? 1.0 - std::exp(-total_rate * dt_local) : 0.0; // Probability a transition happens this timestep

    // Decide if a transition occurs
    if (uniform_dist(local_rng) < p_react && total_rate > 0.0) { // If true, a state change occurs

        // Get build CDF
        std::vector<double> prob_CDF(outgoing_rates.size());
        double prob_accumulation = 0.0;
        for (size_t i = 0; i < outgoing_rates.size(); i++) {
            prob_accumulation += outgoing_rates[i] / total_rate;
            prob_CDF[i] = prob_accumulation;
        }

        // Select Next State
        double r = uniform_dist(local_rng);
        auto it = std::upper_bound(prob_CDF.begin(), prob_CDF.end(), r);
        size_t idx = std::distance(prob_CDF.begin(), it); // We don't want an iterator, buy rather an integer index
        if (idx >= reaction_indices.size()) idx = reaction_indices.size() - 1; // clamp due to floating point rounding
        new_state = reaction_indices[idx];

        // Count full rotations
        const int last_state = static_cast<int>(theta_states.value().size()) - 1;
        if (prev_state == last_state && new_state == 0) delta_cycle = 1;
        else if (prev_state == 0 && new_state == last_state) delta_cycle = -1;
    }

    return { new_state, delta_cycle };
}

double LangevinGillespie::update_theta(int state, int cycle_count) const {
    return theta_states->at(state) + static_cast<double>(cycle_count) * 2.0 * std::numbers::pi / 3.0;
}

double LangevinGillespie::update_angle(double prev_angle, double target_theta, const std::string& method_lowered, std::mt19937& local_rng) const {
    if (method_lowered == "heun")
        return heun_1d(prev_angle, target_theta, local_rng);
    else if (method_lowered == "euler")
        return euler_maruyama(prev_angle, target_theta, local_rng);
    else if (method_lowered == "probabilistic")
        return probabilistic(prev_angle, target_theta, local_rng);

    throw std::runtime_error("Method must be 'heun', 'euler', or 'probabilistic'");
}

double LangevinGillespie::drift(double current_angle, double target_theta) const {
    return -kappa.value() * (current_angle - target_theta) / gammaB.value();
}

double LangevinGillespie::diffusion() const {
    return std::sqrt(2.0 * kBT.value() / gammaB.value());
}

double LangevinGillespie::heun_1d(double current_angle, double theta_target, std::mt19937& local_rng) const {
    double drift_term = drift(current_angle, theta_target);
    double diffusion_term = diffusion();
    double eta = normal_dist(local_rng);

    // Predictor step
    double sqrt_dt = std::sqrt(dt.value());
    double y_predict = current_angle + dt.value() * drift_term + sqrt_dt * diffusion_term * eta;

    // Corrector step
    double drift_predict = drift(y_predict, theta_target);
    return current_angle + (dt.value() / 2.0) * (drift_term + drift_predict) + sqrt_dt * diffusion_term * eta;
}


double LangevinGillespie::euler_maruyama(double current_angle, double theta_target, std::mt19937& local_rng) const {
    double drift_term = drift(current_angle, theta_target);
    double diffusion_term = diffusion() * std::sqrt(dt.value());
    double eta = normal_dist(local_rng);
    return current_angle + dt.value() * drift_term + diffusion_term * eta;
}

double LangevinGillespie::probabilistic(double current_angle, double target_theta, std::mt19937& local_rng) const {
    double exp_factor = std::exp(-kappa.value() / gammaB.value() * dt.value());
    double mean = target_theta + (current_angle - target_theta) * exp_factor;
    double std_dev = std::sqrt(kBT.value() / kappa.value() * (1.0 - (exp_factor * exp_factor)));
    double eta = normal_dist(local_rng);
    return mean + std_dev * eta;
}

LangevinGillespie::LGParams LangevinGillespie::to_struct() const {
    LGParams params{};
    params.steps = static_cast<unsigned int>(steps.value());
    params.dt = static_cast<float>(dt.value());

    // map method string -> int (0=heun,1=euler,2=probabilistic)
    std::string m = method.value();
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return std::tolower(c); });

    params.method = (m == "heun") ? 0 : (m == "euler") ? 1 : 2;
    params.theta_0 = static_cast<float>(theta_0.has_value() ? theta_0.value() : theta_states.value().at(initial_state.value()));
    params.kappa = static_cast<float>(kappa.value());
    params.kBT = static_cast<float>(kBT.value());
    params.gammaB = static_cast<float>(gammaB.value());
    params.initial_state = static_cast<unsigned int>(initial_state.value());

    // Copy theta_states (copy up to params capacity)
    const auto& ts = theta_states.value();
    const size_t n_ts = std::min(ts.size(), std::size(params.theta_states));
    for (size_t i = 0; i < n_ts; ++i) params.theta_states[i] = static_cast<float>(ts[i]);
    for (size_t i = n_ts; i < std::size(params.theta_states); ++i) params.theta_states[i] = 0.0f; // zero any remaining entries

    // Flatten transition_matrix into row-major float array
    const auto& tm = transition_matrix.value();
    const size_t nrows = std::min(tm.size(), static_cast<size_t>(4));
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i < nrows && j < tm[i].size()) params.transition_matrix[i * 4 + j] = static_cast<float>(tm[i][j]);
            else params.transition_matrix[i * 4 + j] = 0.0f;
        }
    }
    return params;
}
