// src/core/cuda/LangevinGillespie.cu
#include "../include/LangevinGillespie.h"

// -=-=-=-=-=-=-=-=-= STD lib -=-=-=-=-=-=-=-=-=
#include <stdexcept> // std::runtime_error

// -=-=-=-=-=-=-=-=-= Cuda -=-=-=-=-=-=-=-=-=
#include <cuda_runtime.h>    // cudaMalloc, cudaMemcpy, cudaFree, cudaHostAlloc, cudaDeviceSynchronize, cudaGetLastError
#include <curand_kernel.h>   // curandState, curand_init, curand_uniform, curand_normal
#include <cmath>             // sqrtf, expf (host-side math)

 // -=-=-=-=-=-=-=-=-= Device Functions -=-=-=-=-=-=-=-=-=
__device__ float drift(float theta, float target_theta, float kappa, float gammaB) {
    return -kappa * (theta - target_theta) / gammaB;
}

__device__ float diffusion(float kBT, float gammaB) {
    return sqrtf(2.0f * kBT / gammaB);
}

__device__ float euler_maruyama(float theta, float target_theta, float dt, float kappa, float kBT, float gammaB, curandState* rng) {
    float d = drift(theta, target_theta, kappa, gammaB);
    float s = diffusion(kBT, gammaB) * sqrtf(dt);
    return theta + dt * d + s * curand_normal(rng);
}

__device__ float heun(float theta, float target_theta, float dt, float kappa, float kBT, float gammaB, curandState* rng) {
    float d = drift(theta, target_theta, kappa, gammaB);
    float s = diffusion(kBT, gammaB);
    float eta = curand_normal(rng);

    float predict = theta + dt * d + sqrtf(dt) * s * eta;
    float d_predict = drift(predict, target_theta, kappa, gammaB);
    return theta + 0.5f * dt * (d + d_predict) + sqrtf(dt) * s * eta;
}

__device__ float probabilistic(float theta, float target_theta, float dt, float kappa, float kBT, float gammaB, curandState* rng) {
    float exp_factor = expf(-kappa / gammaB * dt);
    float mean = target_theta + (theta - target_theta) * exp_factor;
    float std_dev = sqrtf(kBT / kappa * (1.0f - exp_factor * exp_factor));
    return mean + std_dev * curand_normal(rng);
}

// -=-=-=-=-=-=-=-=-= Kernel -=-=-=-=-=-=-=-=-=
extern "C" __global__ void simulate_kernel(const LangevinGillespie::LGParams* params,
    float* bead_positions,
    int* states,
    float* target_thetas,
    int nSim,
    unsigned long long seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nSim) return;

    int steps = static_cast<int>(params->steps);
    curandState rng;
    curand_init(seed + idx, 0, 0, &rng);

    int state = static_cast<int>(params->initial_state);
    float theta = params->theta_0;
    int cycle = 0;

    const float two_pi_over_3 = 2.0f * 3.14159265358979323846f / 3.0f;

    for (int i = 0; i < steps; ++i) {
        float target_theta = params->theta_states[state] + cycle * two_pi_over_3;

        // Function pointer array for method selection (reduces divergence)
        typedef float(*MethodFunc)(float, float, float, float, float, float, curandState*);
        __shared__ MethodFunc methods[3];
        if (threadIdx.x == 0) {
            methods[0] = heun;
            methods[1] = euler_maruyama;
            methods[2] = probabilistic;
        }

        __syncthreads();

        theta = methods[params->method](theta, target_theta, params->dt, params->kappa, params->kBT, params->gammaB, &rng);

        bead_positions[idx * steps + i] = theta;
        target_thetas[idx * steps + i] = target_theta;
        states[idx * steps + i] = state;

        // Compute total transition rate
        float total_rate = 0.0f;

#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (j != state) total_rate += params->transition_matrix[state * 4 + j];

        // Determine if state changes 
        if (total_rate > 0.0f && curand_uniform(&rng) < 1.0f - expf(-total_rate * params->dt)) {
            float r = curand_uniform(&rng);
            float cum = 0.0f;
            int new_state = state;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j == state) continue;
                cum += params->transition_matrix[state * 4 + j] / total_rate;
                if (r <= cum) { new_state = j; break; }
            }

            if (state == 3 && new_state == 0) cycle++;
            else if (state == 0 && new_state == 3) cycle--;
            state = new_state;
        }
    }
}

// -=-=-=-=-=-=-=-=-= Host Simulation Wrapper -=-=-=-=-=-=-=-=-=
std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<float>>
LangevinGillespie::simulate_multithreaded_cuda(int nSim, unsigned long long seed) {
    this->verify_attributes();
    LGParams h_params = this->to_struct();
    size_t steps = static_cast<size_t>(h_params.steps);
    size_t total_elements = nSim * steps;

    py::ssize_t dim0 = nSim;
    py::ssize_t dim1 = steps;

    // Allocate device struct if needed
    if (!d_params) cudaMalloc(&d_params, sizeof(LGParams));
    cudaMemcpy(d_params, &h_params, sizeof(LGParams), cudaMemcpyHostToDevice);

    // Attempt pinned mapped host memory
    float* h_beads = nullptr, * h_thetas = nullptr;
    int* h_states = nullptr;
    float* d_beads_ptr = nullptr, * d_thetas_ptr = nullptr;
    int* d_states_ptr = nullptr;

    bool using_mapped = false;
    if (cudaHostAlloc(&h_beads, total_elements * sizeof(float), cudaHostAllocMapped) == cudaSuccess) {
        cudaHostAlloc(&h_thetas, total_elements * sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc(&h_states, total_elements * sizeof(int), cudaHostAllocMapped);

        cudaHostGetDevicePointer(&d_beads_ptr, h_beads, 0);
        cudaHostGetDevicePointer(&d_thetas_ptr, h_thetas, 0);
        cudaHostGetDevicePointer(&d_states_ptr, h_states, 0);
        using_mapped = true;
    } else {
        if (current_allocated_size < total_elements) {
            if (d_beads) cudaFree(d_beads);
            if (d_thetas) cudaFree(d_thetas);
            if (d_states) cudaFree(d_states);
            cudaMalloc(&d_beads, total_elements * sizeof(float));
            cudaMalloc(&d_thetas, total_elements * sizeof(float));
            cudaMalloc(&d_states, total_elements * sizeof(int));
            current_allocated_size = total_elements;
        }
        d_beads_ptr = d_beads;
        d_thetas_ptr = d_thetas;
        d_states_ptr = d_states;
    }

    // Launch kernel
    const int block_size = 256;
    const int grid_size = (nSim + block_size - 1) / block_size;
    simulate_kernel << <grid_size, block_size >> > (d_params, d_beads_ptr, d_states_ptr, d_thetas_ptr, nSim, seed);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));

    // Wrap as numpy arrays
    if (!using_mapped) {
        std::vector<float> h_beads_vec(total_elements);
        std::vector<float> h_thetas_vec(total_elements);
        std::vector<int> h_states_vec(total_elements);

        cudaMemcpy(h_beads_vec.data(), d_beads, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_thetas_vec.data(), d_thetas, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_states_vec.data(), d_states, total_elements * sizeof(int), cudaMemcpyDeviceToHost);

        py::array_t<float> py_beads({ dim0, dim1 }, h_beads_vec.data());
        py::array_t<float> py_thetas({ dim0, dim1 }, h_thetas_vec.data());
        py::array_t<int> py_states({ dim0, dim1 }, h_states_vec.data());
        return { py_beads, py_states, py_thetas };
    }

    auto free_float_host = [](void* p) { if (p) cudaFreeHost(p); };
    auto free_int_host = [](void* p) { if (p) cudaFreeHost(p); };

    py::capsule beads_capsule(h_beads, free_float_host);
    py::capsule thetas_capsule(h_thetas, free_float_host);
    py::capsule states_capsule(h_states, free_int_host);

    py::array py_beads(py::buffer_info(h_beads, sizeof(float), py::format_descriptor<float>::format(),
        2, { dim0, dim1 }, { dim1 * sizeof(float), sizeof(float) }), beads_capsule);
    py::array py_thetas(py::buffer_info(h_thetas, sizeof(float), py::format_descriptor<float>::format(),
        2, { dim0, dim1 }, { dim1 * sizeof(float), sizeof(float) }), thetas_capsule);
    py::array py_states(py::buffer_info(h_states, sizeof(int), py::format_descriptor<int>::format(),
        2, { dim0, dim1 }, { dim1 * sizeof(int), sizeof(int) }), states_capsule);

    return { py_beads.cast<py::array_t<float>>(), py_states.cast<py::array_t<int>>(), py_thetas.cast<py::array_t<float>>() };
}
