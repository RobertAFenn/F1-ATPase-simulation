// src/core/cuda/LangevinGillespie_opt.cu
#include "../include/LangevinGillespie.h"
#include <stdexcept>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <string>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e)); } while(0)

// Device inline helpers (same math as yours)
__device__ inline double drift_f(double theta, double target_theta, double kappa, double gammaB) {
    return -kappa * (theta - target_theta) / gammaB;
}
__device__ inline double diffusion_f(double kBT, double gammaB) {
    return sqrtf(2.0f * kBT / gammaB);
}
__device__ inline double euler_step(double theta, double target_theta, double dt, double kappa, double kBT, double gammaB, curandStatePhilox4_32_10_t &rng) {
    double d = drift_f(theta, target_theta, kappa, gammaB);
    double s = diffusion_f(kBT, gammaB) * sqrtf(dt);
    double z = curand_normal(&rng);
    return theta + dt * d + s * z;
}
__device__ inline double heun_step(double theta, double target_theta, double dt, double kappa, double kBT, double gammaB, curandStatePhilox4_32_10_t &rng) {
    double d = drift_f(theta, target_theta, kappa, gammaB);
    double s = diffusion_f(kBT, gammaB);
    double eta = curand_normal(&rng);
    double predict = theta + dt * d + sqrtf(dt) * s * eta;
    double d_predict = drift_f(predict, target_theta, kappa, gammaB);
    return theta + 0.5f * dt * (d + d_predict) + sqrtf(dt) * s * eta;
}
__device__ inline double prob_step(double theta, double target_theta, double dt, double kappa, double kBT, double gammaB, curandStatePhilox4_32_10_t &rng) {
    double exp_factor = expf(-kappa / gammaB * dt);
    double mean = target_theta + (theta - target_theta) * exp_factor;
    double std_dev = sqrtf(kBT / kappa * (1.0f - exp_factor * exp_factor));
    return mean + std_dev * curand_normal(&rng);
}

// Common kernel pattern: each kernel implements one integrator so it can be inlined.
extern "C" __global__ void simulate_kernel_euler(const LangevinGillespie::LGParams* params,
    double* bead_positions_step_major, // layout: [step * nSim + sim]
    int* states_step_major,
    double* target_thetas_step_major,
    int nSim,
    unsigned long long seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nSim) return;

    // Load params into registers (local variables) to avoid repeated global reads
    const int steps = static_cast<int>(params->steps);
    const double dt = params->dt;
    const double kappa = params->kappa;
    const double kBT = params->kBT;
    const double gammaB = params->gammaB;
    const int initial_state = static_cast<int>(params->initial_state);
    const double theta_0 = params->theta_0;
    const double* theta_states = params->theta_states;
    const double two_pi_over_3 = 2.0f * 3.14159265358979323846f / 3.0f;
    const double* trans = params->transition_matrix;

    curandStatePhilox4_32_10_t rng;
    curand_init((unsigned long long)seed, (unsigned long long)idx, 0, &rng);

    int state = initial_state;
    double theta = theta_0;
    int cycle = 0;

    for (int i = 0; i < steps; ++i) {
        double target_theta = theta_states[state] + cycle * two_pi_over_3;

        // step (Euler)
        theta = euler_step(theta, target_theta, dt, kappa, kBT, gammaB, rng);

        // coalesced write: step-major layout so threads write adjacent addresses
        int base = i * nSim + idx;
        bead_positions_step_major[base] = theta;
        target_thetas_step_major[base] = target_theta;
        states_step_major[base] = state;

        // Gillespie-like switching per-step
        double total_rate = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) if (j != state) total_rate += trans[state * 4 + j];

        if (total_rate > 0.0f) {
            double p_jump = 1.0f - expf(-total_rate * dt);
            if (curand_uniform(&rng) < p_jump) {
                double r = curand_uniform(&rng);
                double cum = 0.0f;
                int new_state = state;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j == state) continue;
                    cum += trans[state * 4 + j] / total_rate;
                    if (r <= cum) { new_state = j; break; }
                }
                if (state == 3 && new_state == 0) cycle++;
                else if (state == 0 && new_state == 3) cycle--;
                state = new_state;
            }
        }
    }
}

extern "C" __global__ void simulate_kernel_heun(const LangevinGillespie::LGParams* params,
    double* bead_positions_step_major,
    int* states_step_major,
    double* target_thetas_step_major,
    int nSim,
    unsigned long long seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nSim) return;

    const int steps = static_cast<int>(params->steps);
    const double dt = params->dt;
    const double kappa = params->kappa;
    const double kBT = params->kBT;
    const double gammaB = params->gammaB;
    const int initial_state = static_cast<int>(params->initial_state);
    const double theta_0 = params->theta_0;
    const double* theta_states = params->theta_states;
    const double two_pi_over_3 = 2.0f * 3.14159265358979323846f / 3.0f;
    const double* trans = params->transition_matrix;

    curandStatePhilox4_32_10_t rng;
    curand_init((unsigned long long)seed, (unsigned long long)idx, 0, &rng);

    int state = initial_state;
    double theta = theta_0;
    int cycle = 0;

    for (int i = 0; i < steps; ++i) {
        double target_theta = theta_states[state] + cycle * two_pi_over_3;

        theta = heun_step(theta, target_theta, dt, kappa, kBT, gammaB, rng);

        int base = i * nSim + idx;
        bead_positions_step_major[base] = theta;
        target_thetas_step_major[base] = target_theta;
        states_step_major[base] = state;

        double total_rate = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) if (j != state) total_rate += trans[state * 4 + j];

        if (total_rate > 0.0f) {
            double p_jump = 1.0f - expf(-total_rate * dt);
            if (curand_uniform(&rng) < p_jump) {
                double r = curand_uniform(&rng);
                double cum = 0.0f;
                int new_state = state;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j == state) continue;
                    cum += trans[state * 4 + j] / total_rate;
                    if (r <= cum) { new_state = j; break; }
                }
                if (state == 3 && new_state == 0) cycle++;
                else if (state == 0 && new_state == 3) cycle--;
                state = new_state;
            }
        }
    }
}

extern "C" __global__ void simulate_kernel_prob(const LangevinGillespie::LGParams* params,
    double* bead_positions_step_major,
    int* states_step_major,
    double* target_thetas_step_major,
    int nSim,
    unsigned long long seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nSim) return;

    const int steps = static_cast<int>(params->steps);
    const double dt = params->dt;
    const double kappa = params->kappa;
    const double kBT = params->kBT;
    const double gammaB = params->gammaB;
    const int initial_state = static_cast<int>(params->initial_state);
    const double theta_0 = params->theta_0;
    const double* theta_states = params->theta_states;
    const double two_pi_over_3 = 2.0 * 3.14159265358979323846 / 3.0; 
    const double* trans = params->transition_matrix;

    curandStatePhilox4_32_10_t rng;
    curand_init((unsigned long long)seed, (unsigned long long)idx, 0, &rng);

    int state = initial_state;
    double theta = theta_0;
    int cycle = 0;

    for (int i = 0; i < steps; ++i) {
        double target_theta = theta_states[state] + cycle * two_pi_over_3;

        theta = prob_step(theta, target_theta, dt, kappa, kBT, gammaB, rng);

        int base = i * nSim + idx;
        bead_positions_step_major[base] = theta;
        target_thetas_step_major[base] = target_theta;
        states_step_major[base] = state;

        double total_rate = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) if (j != state) total_rate += trans[state * 4 + j];

        if (total_rate > 0.0f) {
            double p_jump = 1.0f - expf(-total_rate * dt);
            if (curand_uniform(&rng) < p_jump) {
                double r = curand_uniform(&rng);
                double cum = 0.0f;
                int new_state = state;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j == state) continue;
                    cum += trans[state * 4 + j] / total_rate;
                    if (r <= cum) { new_state = j; break; }
                }
                if (state == 3 && new_state == 0) cycle++;
                else if (state == 0 && new_state == 3) cycle--;
                state = new_state;
            }
        }
    }
}

std::tuple<py::array_t<double>, py::array_t<int>, py::array_t<double>>
LangevinGillespie::simulate_multithreaded_cuda(unsigned int nSim, unsigned long long seed) {
    this->verify_attributes();
    if (nSim < 1) throw std::invalid_argument("nSim must be greater than 0!");

    LGParams h_params = this->to_struct();
    size_t steps = static_cast<size_t>(h_params.steps);
    size_t total_elements = (size_t)nSim * steps;

    py::ssize_t dim0 = nSim;
    py::ssize_t dim1 = steps;

    // copy params to device
    if (!d_params) CUDA_CHECK(cudaMalloc(&d_params, sizeof(LGParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &h_params, sizeof(LGParams), cudaMemcpyHostToDevice));

    // allocate device buffers (step-major layout: [step * nSim + sim])
    double* d_beads = nullptr;
    double* d_thetas = nullptr;
    int* d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_beads, total_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_thetas, total_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_states, total_elements * sizeof(int)));

    // allocate pinned host buffers for copy target (faster async copies)
    double* h_beads = nullptr;
    double* h_thetas = nullptr;
    int* h_states = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&h_beads, total_elements * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_thetas, total_elements * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_states, total_elements * sizeof(int), cudaHostAllocDefault));

    // launch kernel
    const int block_size = 256;
    const int grid_size = (nSim + block_size - 1) / block_size;

    // choose kernel by method (0=euler,1=heun,2=prob)
    int method = static_cast<int>(h_params.method);
    if (method == 0) simulate_kernel_euler<<<grid_size, block_size>>>(d_params, d_beads, d_states, d_thetas, nSim, seed);
    else if (method == 1) simulate_kernel_heun<<<grid_size, block_size>>>(d_params, d_beads, d_states, d_thetas, nSim, seed);
    else simulate_kernel_prob<<<grid_size, block_size>>>(d_params, d_beads, d_states, d_thetas, nSim, seed);

    // async copy device -> pinned host and synchronize
    CUDA_CHECK(cudaMemcpyAsync(h_beads, d_beads, total_elements * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(h_thetas, d_thetas, total_elements * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(h_states, d_states, total_elements * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));

    // Build numpy arrays WITHOUT additional copy by setting strides so logical (nSim, steps)
    // buffer layout is step-major: index = step * nSim + sim
    // so element (sim, step) in logical array is at offset: step * nSim + sim
    // therefore strides are: sim_stride = sizeof(element), step_stride = nSim * sizeof(element)
    auto free_double_host = [](void* p) { if (p) cudaFreeHost(p); };
    auto free_int_host = [](void* p) { if (p) cudaFreeHost(p); };

    py::capsule beads_capsule(h_beads, free_double_host);
    py::capsule thetas_capsule(h_thetas, free_double_host);
    py::capsule states_capsule(h_states, free_int_host);

    // double buffer_info: ptr, itemsize, format, ndim, shape, strides
    py::array py_beads(py::buffer_info(h_beads, sizeof(double), py::format_descriptor<double>::format(),
        2, { dim0, dim1 }, { sizeof(double), dim0 * sizeof(double) }), beads_capsule);
    py::array py_thetas(py::buffer_info(h_thetas, sizeof(double), py::format_descriptor<double>::format(),
        2, { dim0, dim1 }, { sizeof(double), dim0 * sizeof(double) }), thetas_capsule);
    py::array py_states(py::buffer_info(h_states, sizeof(int), py::format_descriptor<int>::format(),
        2, { dim0, dim1 }, { sizeof(int), dim0 * sizeof(int) }), states_capsule);

    // free device buffers (keep host pinned buffers for Python's ownership)
    CUDA_CHECK(cudaFree(d_beads));
    CUDA_CHECK(cudaFree(d_thetas));
    CUDA_CHECK(cudaFree(d_states));

    return { py_beads.cast<py::array_t<double>>(), py_states.cast<py::array_t<int>>(), py_thetas.cast<py::array_t<double>>() };
}
