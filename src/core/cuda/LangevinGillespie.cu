// src/core/cuda/LangevinGillespie.cu
#include "../include/LangevinGillespie.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <iostream>

namespace py = pybind11;

// -=-=-=-=-=-=-=-=-= Device RNG -=-=-=-=-=-=-=-=-=
__device__ float normal(curandState* state) {
    return curand_normal(state); // standard normal ~ N(0,1)
}

// -=-=-=-=-=-=-=-=-= Device Functions -=-=-=-=-=-=-=-=-=
__device__ float drift_dev(float current_angle, float target_theta, float kappa, float gammaB) {
    return -kappa * (current_angle - target_theta) / gammaB;
}

__device__ float diffusion_dev(float kBT, float gammaB) {
    return sqrtf(2.0f * kBT / gammaB);
}

__device__ float euler_maruyama_dev(float current_angle, float target_theta,
    float dt, float kappa, float kBT, float gammaB,
    curandState* rng) {
    float drift_term = drift_dev(current_angle, target_theta, kappa, gammaB);
    float diffusion_term = diffusion_dev(kBT, gammaB) * sqrtf(dt);
    float eta = normal(rng);
    return current_angle + dt * drift_term + diffusion_term * eta;
}

__device__ float heun_1d_dev(float current_angle, float target_theta,
    float dt, float kappa, float kBT, float gammaB,
    curandState* rng) {
    float drift_term = drift_dev(current_angle, target_theta, kappa, gammaB);
    float diffusion_term = diffusion_dev(kBT, gammaB);
    float eta = normal(rng);

    float y_predict = current_angle + dt * drift_term + sqrtf(dt) * diffusion_term * eta;
    float drift_predict = drift_dev(y_predict, target_theta, kappa, gammaB);

    return current_angle + 0.5f * dt * (drift_term + drift_predict) + sqrtf(dt) * diffusion_term * eta;
}

__device__ float probabilistic_dev(float current_angle, float target_theta,
    float dt, float kappa, float kBT, float gammaB,
    curandState* rng) {

    float exp_factor = expf(-kappa / gammaB * dt);
    float mean = target_theta + (current_angle - target_theta) * exp_factor;
    float std_dev = sqrtf(kBT / kappa * (1.0f - exp_factor * exp_factor));
    float eta = normal(rng);
    return mean + std_dev * eta;
}

// -=-=-=-=-=-=-=-=-= Kernel -=-=-=-=-=-=-=-=-=
extern "C" __global__ void simulate_kernel(const LangevinGillespie::LGParams* params,
    float* bead_positions,
    int* states,
    float* target_thetas,
    int nSim,
    unsigned long long seed,
    int threads_per_sim) {

    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sim_idx >= nSim) return;

    int steps = static_cast<int>(params->steps);

    // Initialize RNG per simulation
    curandState rng_state;
    curand_init(static_cast<unsigned long long>(seed) + sim_idx, 0, 0, &rng_state);

    // Initial conditions
    int state = static_cast<int>(params->initial_state);
    float theta = params->theta_0;
    int cycle_count = 0;

    for (int i = 0; i < steps; ++i) {
        // Compute target theta (2*pi/3 per full 4-state rotation)
        float target_theta = params->theta_states[state] + cycle_count * 2.0f * 3.14159265358979323846f / 3.0f;

        // Update angle based on method
        if (params->method == 0)
            theta = heun_1d_dev(theta, target_theta, params->dt, params->kappa, params->kBT, params->gammaB, &rng_state);
        else if (params->method == 1)
            theta = euler_maruyama_dev(theta, target_theta, params->dt, params->kappa, params->kBT, params->gammaB, &rng_state);
        else // Probabilistic
            theta = probabilistic_dev(theta, target_theta, params->dt, params->kappa, params->kBT, params->gammaB, &rng_state);

        // Save results
        bead_positions[sim_idx * steps + i] = theta;
        target_thetas[sim_idx * steps + i] = target_theta;
        states[sim_idx * steps + i] = state;

        // Sample next state (4-state flatten)
        float total_rate = 0.0f;
        for (int j = 0; j < 4; ++j) {
            if (j != state)
                total_rate += params->transition_matrix[state * 4 + j];
        }

        float p_react = 1.0f - expf(-total_rate * params->dt);

        if (curand_uniform(&rng_state) < p_react && total_rate > 0.0f) {
            float r = curand_uniform(&rng_state);
            float cumulative = 0.0f;
            int new_state = state;
            for (int j = 0; j < 4; ++j) {
                if (j == state) continue;
                cumulative += params->transition_matrix[state * 4 + j] / total_rate;
                if (r <= cumulative) {
                    new_state = j;
                    break;
                }
            }

            // Count full rotations
            if (state == 3 && new_state == 0)
                cycle_count += 1;
            else if (state == 0 && new_state == 3)
                cycle_count -= 1;

            state = new_state;
        }
    }
}



std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<float>>
LangevinGillespie::simulate_multithreaded_cuda(int nSim, unsigned long long seed) {
    this->verify_attributes();

    // host params -> device params
    LGParams h_params = this->to_struct();
    const size_t steps = static_cast<size_t>(h_params.steps);
    const size_t total_elements = static_cast<size_t>(nSim) * steps;

    // numpy shape dims
    py::ssize_t dim0 = static_cast<py::ssize_t>(nSim);
    py::ssize_t dim1 = static_cast<py::ssize_t>(steps);

    // Allocate device params struct (persistent member)
    if (!d_params) {
        cudaError_t e = cudaMalloc(&d_params, sizeof(LGParams));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc d_params failed: ") + cudaGetErrorString(e));
    }
    cudaMemcpy(d_params, &h_params, sizeof(LGParams), cudaMemcpyHostToDevice);

    // Try pinned mapped host memory for zero copy
    float* h_beads_host = nullptr;
    float* h_thetas_host = nullptr;
    int* h_states_host = nullptr;
    float* d_beads_mapped = nullptr;
    float* d_thetas_mapped = nullptr;
    int* d_states_mapped = nullptr;

    bool using_mapped = false;

    // Allocate mapped host buffers (zero-copy) - best performance when available
    cudaError_t herr = cudaHostAlloc(reinterpret_cast<void**>(&h_beads_host), total_elements * sizeof(float), cudaHostAllocMapped);
    if (herr == cudaSuccess) {
        herr = cudaHostAlloc(reinterpret_cast<void**>(&h_thetas_host), total_elements * sizeof(float), cudaHostAllocMapped);
        herr = cudaHostAlloc(reinterpret_cast<void**>(&h_states_host), total_elements * sizeof(int), cudaHostAllocMapped);
        // get device pointers to mapped host buffers (kernel will use these)
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_beads_mapped), h_beads_host, 0);
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_thetas_mapped), h_thetas_host, 0);
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_states_mapped), h_states_host, 0);
        using_mapped = true;
    } else {
        // fallback: allocate device buffers (persistent members)
        if (current_allocated_size < total_elements) {
            if (d_beads) cudaFree(d_beads);
            if (d_thetas) cudaFree(d_thetas);
            if (d_states) cudaFree(d_states);

            cudaError_t e1 = cudaMalloc(&d_beads, total_elements * sizeof(float));
            cudaError_t e2 = cudaMalloc(&d_thetas, total_elements * sizeof(float));
            cudaError_t e3 = cudaMalloc(&d_states, total_elements * sizeof(int));
            if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc device buffers failed: ") + cudaGetErrorString(e1));
            }
            current_allocated_size = total_elements;
        }
    }

    // Kernel launch config
    const int block_size = 256;
    const int grid_size = static_cast<int>((nSim + block_size - 1) / block_size);
    const int threads_per_sim = 1;

    // Choose pointers for kernel (mapped or device)
    float* kernel_beads_ptr = using_mapped ? d_beads_mapped : d_beads;
    float* kernel_thetas_ptr = using_mapped ? d_thetas_mapped : d_thetas;
    int* kernel_states_ptr = using_mapped ? d_states_mapped : d_states;

    // Launch kernel 
    simulate_kernel << <grid_size, block_size >> > (d_params, kernel_beads_ptr, kernel_states_ptr, kernel_thetas_ptr, nSim, seed, threads_per_sim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    cudaDeviceSynchronize();

    // If not using mapped memory, copy device -> host into numpy arrays
    if (!using_mapped) {
        // allocate host vectors and copy back
        std::vector<float> h_beads(total_elements);
        std::vector<float> h_thetas(total_elements);
        std::vector<int>   h_states(total_elements);

        cudaMemcpy(h_beads.data(), d_beads, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_thetas.data(), d_thetas, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_states.data(), d_states, total_elements * sizeof(int), cudaMemcpyDeviceToHost);

        // wrap as numpy arrays (copies)
        py::array_t<float> py_beads({ dim0, dim1 });
        py::array_t<float> py_thetas({ dim0, dim1 });
        py::array_t<int>   py_states({ dim0, dim1 });

        std::memcpy(py_beads.mutable_data(), h_beads.data(), total_elements * sizeof(float));
        std::memcpy(py_thetas.mutable_data(), h_thetas.data(), total_elements * sizeof(float));
        std::memcpy(py_states.mutable_data(), h_states.data(), total_elements * sizeof(int));

        return { py_beads, py_states, py_thetas };
    }

    // Using mapped host memory: create capsules so Python will free with cudaFreeHost
    auto free_float_host = [](void* p) {
        if (p) cudaFreeHost(p);
        };
    auto free_int_host = [](void* p) {
        if (p) cudaFreeHost(p);
        };

    py::capsule beads_capsule(h_beads_host, free_float_host);
    py::capsule thetas_capsule(h_thetas_host, free_float_host);
    py::capsule states_capsule(h_states_host, free_int_host);

    // Build buffer_info for py::array that wraps the mapped host memory (row-major)
    py::buffer_info beads_info(
        h_beads_host,
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        { dim0, dim1 },
        { dim1 * sizeof(float), sizeof(float) }
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

    py::array py_beads(beads_info, beads_capsule);
    py::array py_thetas(thetas_info, thetas_capsule);
    py::array py_states(states_info, states_capsule);

    return { py_beads.cast<py::array_t<float>>(), py_states.cast<py::array_t<int>>(), py_thetas.cast<py::array_t<float>>() };
}
