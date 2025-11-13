#include "LangevinGillespie.h"
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <stdexcept>

// Optional CUDA error check macro
#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) \
        throw std::runtime_error(cudaGetErrorString(err)); \
} while(0)

// Example kernel
__global__ void LangevinKernel(
    float* beads, int* states, float* thetas,
    LangevinGillespie::LGParams params,
    unsigned long long seed, int nSim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nSim) return;
  
}

std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<float>>
LangevinGillespie::simulate_multithreaded_cuda(int nSim, unsigned long long base_seed)
{
    verify_attributes();               // from .cpp file
    LGParams params = to_struct();     // from .cpp file

    // --- Device allocations ---
    size_t total_elements = static_cast<size_t>(params.steps) * nSim;
    float *d_beads = nullptr, *d_thetas = nullptr;
    int   *d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_beads, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_thetas, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_states, total_elements * sizeof(int)));

    // --- Launch kernel ---
    const int block_size = 256;
    const int grid_size = (nSim + block_size - 1) / block_size;
    LangevinKernel<<<grid_size, block_size>>>(d_beads, d_states, d_thetas, params, base_seed, nSim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Copy results back ---
    std::vector<float> h_beads(total_elements);
    std::vector<float> h_thetas(total_elements);
    std::vector<int>   h_states(total_elements);
    CUDA_CHECK(cudaMemcpy(h_beads.data(), d_beads, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_thetas.data(), d_thetas, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_states.data(), d_states, total_elements * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_beads));
    CUDA_CHECK(cudaFree(d_thetas));
    CUDA_CHECK(cudaFree(d_states));

    // --- Return as numpy arrays ---
    return {
        py::array_t<float>(h_beads.size(), h_beads.data()),
        py::array_t<int>(h_states.size(), h_states.data()),
        py::array_t<float>(h_thetas.size(), h_thetas.data())
    };
}
