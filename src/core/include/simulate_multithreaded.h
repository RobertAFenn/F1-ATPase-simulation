#pragma once
#include "LangevinGillespie.h"
#include <cuda_runtime.h>

__global__ void simulate_kernel(
    const LangevinGillespie::LGParams* params,
    float* bead_positions_out,
    int* states_out,
    float* target_theta_out,
    int nSim,
    unsigned long long base_seed,
    int sim_per_thread
);
