#include "../include/LangevinGillespie.h"
#include "../include/simulate_multithreaded.h"
#include <algorithm>      // for std::min, std::max, std::clamp, std::find, etc.
#include <cctype>         // for std::tolower (if you normalize "method")
#include <iostream>       
#include <numeric>        // for std::accumulate (if you sum transition rates)
#include <numbers>        // math constants
#include <cuda_runtime.h>   // Core CUDA runtime API
#include <device_launch_parameters.h> // For threadIdx, blockIdx, etc.



// --------------------- Pybind11 Wrap ---------------------

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

        .def("simulate_multithreaded", &LangevinGillespie::simulate_multithreaded,
            py::arg("nSim"),
            py::arg("num_threads"),
            py::arg("seed") = std::nullopt,
            "Run multiple Langevin simulations in parallel.\n"
            "Returns:\n"
            "    bead_positions: list of lists, each inner list is one simulation\n"
            "    states: list of lists, each inner list is discrete states per simulation\n"
            "    target_thetas: list of lists, each inner list is target angles per simulation")


        .def("simulate_multithreaded_cuda", &LangevinGillespie::simulate_multithreaded_cuda,
            py::arg("nSim"),
            py::arg("base_seed") = 1234ULL,
            "Run multiple Langevin simulations on GPU via CUDA.")


        .def("computeGammaB", &LangevinGillespie::computeGammaB,
            py::arg("a") = 0,
            py::arg("r") = 0,
            py::arg("eta") = 0,
            "Compute rotational friction coefficient of the bead.");
}

// --------------------- Public Methods ---------------------
std::tuple<std::vector<double>, std::vector<int>, std::vector<double>>
LangevinGillespie::simulate(const std::optional<unsigned int>& seed) {
    // Step -3: Verify attributes
    verify_attributes();

    // Step -2: Setup RNG
    std::mt19937 rand_gen = create_rng(seed);

    // Step -1: Method cleaning
    std::string method_lowered = method.value();
    std::transform(method_lowered.begin(), method_lowered.end(), method_lowered.begin(),
        [](unsigned char c) { return std::tolower(c); });

    // Step 0: Starting angle
    double theta_start = theta_0.has_value() ? theta_0.value() : theta_states.value()[initial_state.value()];

    if (theta_0.has_value()) {
        double target = theta_states.value()[initial_state.value()];
        double rel_tol = 1e-6;
        if (std::abs(theta_0.value() - target) > rel_tol * std::abs(target)) {
            std::cerr << "Warning: theta_0 (" << theta_0.value()
                << ") does not match the target angle for initial state ("
                << target << "). Simulation will start at theta_0.\n";
        }
    }

    // Step 1: Allocate vectors
    std::vector<double> bead_positions(steps.value());
    std::vector<int> states(steps.value());
    std::vector<double> target_thetas(steps.value());

    // Step 2: Initialize first values
    bead_positions[0] = theta_start;
    states[0] = initial_state.value();
    target_thetas[0] = theta_states.value()[initial_state.value()];
    unsigned int cycle_count = 0;

    // Step 3: Main loop
    for (size_t i = 1; i < steps.value(); ++i) {
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

    // Pre-allocation
    std::vector<std::vector<double>> combined_bead_positions(nSim);
    std::vector<std::vector<int>> combined_states(nSim);
    std::vector<std::vector<double>> combined_target_thetas(nSim);

    // Divide work
    std::vector<std::thread> threads(num_threads);
    unsigned int work_per_thread = nSim / num_threads;
    unsigned int remainder = nSim % num_threads;
    unsigned int start_idx = 0;

    py::gil_scoped_release release;

    for (unsigned int tid = 0; tid < num_threads; ++tid) {
        unsigned int sims_for_this_thread = work_per_thread + (tid < remainder ? 1 : 0);
        unsigned int thread_start_idx = start_idx;
        start_idx += sims_for_this_thread;

        threads[tid] = std::thread([&, sims_for_this_thread, thread_start_idx, seed]() {
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

// TODO Add real "randomization"
std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<float>>
LangevinGillespie::simulate_multithreaded_cuda(int nSim, unsigned long long base_seed) {
    this->verify_attributes();

    // host params -> device params
    LGParams h_params = this->to_struct();
    size_t steps = static_cast<size_t>(h_params.steps);
    size_t total_elements = static_cast<size_t>(nSim) * steps;

    // Make dimensions for numpy
    py::ssize_t dim0 = static_cast<py::ssize_t>(nSim);
    py::ssize_t dim1 = static_cast<py::ssize_t>(steps);

    // Allocate device params struct (unchanged)
    if (!d_params) {
        cudaError_t e = cudaMalloc(&d_params, sizeof(LGParams));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc d_params failed: ") + cudaGetErrorString(e));
    }
    cudaMemcpy(d_params, &h_params, sizeof(LGParams), cudaMemcpyHostToDevice);

    // First, try to allocate mapped (pinned) host memory for zero-copy:
    float* h_beads_host = nullptr;
    float* h_thetas_host = nullptr;
    int* h_states_host = nullptr;
    float* d_beads_mapped = nullptr;
    float* d_thetas_mapped = nullptr;
    int* d_states_mapped = nullptr;

    bool using_mapped = false;

    cudaError_t herr = cudaErrorUnknown;
    herr = cudaHostAlloc(reinterpret_cast<void**>(&h_beads_host), total_elements * sizeof(float), cudaHostAllocMapped);

    if (herr == cudaSuccess) {
        herr = cudaHostAlloc(reinterpret_cast<void**>(&h_thetas_host), total_elements * sizeof(float), cudaHostAllocMapped);
        // Get device pointers that map to the host buffers (kernel will use these)
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_beads_mapped), h_beads_host, 0);
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_thetas_mapped), h_thetas_host, 0);
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_states_mapped), h_states_host, 0);
        using_mapped = true;
    } else {
        // fallback: mapped failed -> free any partial mapped allocations and use device buffers (existing code path)
        if (h_beads_host) cudaFreeHost(h_beads_host);
        if (h_thetas_host) cudaFreeHost(h_thetas_host);
        if (h_states_host) cudaFreeHost(h_states_host);

        // Ensure device buffers exist and are the size we need (reuse your existing members)
        if (current_allocated_size < total_elements) {
            if (d_beads) cudaFree(d_beads);
            if (d_thetas) cudaFree(d_thetas);
            if (d_states) cudaFree(d_states);

            cudaMalloc(&d_beads, total_elements * sizeof(float));
            cudaMalloc(&d_thetas, total_elements * sizeof(float));
            cudaMalloc(&d_states, total_elements * sizeof(int));

            current_allocated_size = total_elements;
        }
    }

    // Kernel launch config
    int block_size = 256;
    int grid_size = (nSim + block_size - 1) / block_size;
    int threads_per_sim = 1;

    // Choose kernel pointers depending on mapped vs device buffers
    float* kernel_beads_ptr = using_mapped ? d_beads_mapped : d_beads;
    float* kernel_thetas_ptr = using_mapped ? d_thetas_mapped : d_thetas;
    int* kernel_states_ptr = using_mapped ? d_states_mapped : d_states;

    // Launch kernel
    simulate_kernel << <grid_size, block_size >> > (d_params, kernel_beads_ptr, kernel_states_ptr, kernel_thetas_ptr, nSim, base_seed, threads_per_sim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));

    cudaDeviceSynchronize();

    // If not using mapped memory, copy device -> host as before
    if (!using_mapped) {
        // allocate host std::vectors and copy back
        std::vector<float> h_beads(total_elements);
        std::vector<float> h_thetas(total_elements);
        std::vector<int>   h_states(total_elements);

        cudaMemcpy(h_beads.data(), d_beads, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_thetas.data(), d_thetas, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_states.data(), d_states, total_elements * sizeof(int), cudaMemcpyDeviceToHost);

        // Wrap into numpy arrays (copy into py arrays) — this costs a memcpy but it's the fallback
        py::array_t<float> py_beads({ dim0, dim1 });
        py::array_t<float> py_thetas({ dim0, dim1 });
        py::array_t<int>   py_states({ dim0, dim1 });

        std::memcpy(py_beads.mutable_data(), h_beads.data(), total_elements * sizeof(float));
        std::memcpy(py_thetas.mutable_data(), h_thetas.data(), total_elements * sizeof(float));
        std::memcpy(py_states.mutable_data(), h_states.data(), total_elements * sizeof(int));

        return { py_beads, py_states, py_thetas };
    }

    // --- USING MAPPED HOST MEMORY: create numpy arrays that *wrap* host ptrs with capsule ---
    // Create capsules that call cudaFreeHost when Python frees them
    auto free_float_host = [](void* p) {
        if (p) cudaFreeHost(p);
        };
    auto free_int_host = [](void* p) {
        if (p) cudaFreeHost(p);
        };

    py::capsule beads_capsule(h_beads_host, free_float_host);
    py::capsule thetas_capsule(h_thetas_host, free_float_host);
    py::capsule states_capsule(h_states_host, free_int_host);

    // Build buffer_info describing the memory layout (row-major: [nSim, steps])
    py::buffer_info beads_info(
        h_beads_host,                             // void* ptr
        sizeof(float),                            // itemsize
        py::format_descriptor<float>::format(),   // format
        2,                                        // ndim
        { dim0, dim1 },                           // shape
        { dim1 * sizeof(float), sizeof(float) }   // strides
    );

    py::buffer_info thetas_info(
        h_thetas_host,
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        { dim0, dim1 },
        { dim1 * sizeof(float), sizeof(float) }
    );

    py::buffer_info states_info(
        h_states_host,
        sizeof(int),
        py::format_descriptor<int>::format(),
        2,
        { dim0, dim1 },
        { dim1 * sizeof(int), sizeof(int) }
    );

    // Create py::array objects that reference the host buffers, with capsule as base (so freeing calls cudaFreeHost)
    py::array py_beads(beads_info, beads_capsule);
    py::array py_thetas(thetas_info, thetas_capsule);
    py::array py_states(states_info, states_capsule);

    return { py_beads.cast<py::array_t<float>>(), py_states.cast<py::array_t<int>>(), py_thetas.cast<py::array_t<float>>() };
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


    throw std::runtime_error("Method must be 'heun', 'euler', or 'probabilistic'");


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

// Replace your existing LangevinGillespie::to_struct() with this.

LangevinGillespie::LGParams LangevinGillespie::to_struct() const {
    // Make sure required attributes are present
    // (caller may already have called verify_attributes())
    // but we be defensive here:
    if (!steps.has_value() || !dt.has_value() || !method.has_value() ||
        !kappa.has_value() || !kBT.has_value() || !gammaB.has_value() ||
        !initial_state.has_value() || !theta_states.has_value() || !transition_matrix.has_value()) {
        throw std::runtime_error("to_struct(): missing required attributes");
    }

    LGParams params;

    // sizes and basic conversions
    params.steps = static_cast<unsigned int>(steps.value());
    params.dt = static_cast<float>(dt.value());

    // map method string -> int (0=heun,1=euler,2=probabilistic)
    std::string m = method.value();
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return std::tolower(c); });
    params.method = (m == "heun") ? 0 : (m == "euler") ? 1 : 2;

    // theta_0: use provided value if present, otherwise default to the theta for initial_state
    if (theta_0.has_value()) {
        params.theta_0 = static_cast<float>(theta_0.value());
    } else {
        // safe because verify_attributes ensures theta_states & initial_state exist
        params.theta_0 = static_cast<float>(theta_states.value().at(initial_state.value()));
    }

    params.kappa = static_cast<float>(kappa.value());
    params.kBT = static_cast<float>(kBT.value());
    params.gammaB = static_cast<float>(gammaB.value());

    params.initial_state = static_cast<unsigned int>(initial_state.value());

    // Copy theta_states (assumes exactly 4 states)
    for (size_t i = 0; i < 4; ++i) {
        params.theta_states[i] = static_cast<float>(theta_states.value().at(i));
    }

    // Flatten transition_matrix into row-major float array
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            params.transition_matrix[i * 4 + j] = static_cast<float>(transition_matrix.value().at(i).at(j));
        }
    }

    return params;
}
