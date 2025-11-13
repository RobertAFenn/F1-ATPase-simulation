#include "../include/simulate_multithreaded.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <math_constants.h> 

namespace cg = cooperative_groups;

// ------------------ GPU Kernel ------------------
__global__ void simulate_kernel(
    const LangevinGillespie::LGParams* __restrict__ params,
    float* __restrict__ bead_positions_out,
    int* __restrict__ states_out,
    float* __restrict__ target_theta_out,
    int nSim,
    unsigned long long base_seed,
    int sim_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_sim = tid * sim_per_thread;

    // init curand (Philox)
    curandStatePhilox4_32_10_t rng;
    curand_init(base_seed, tid, 0, &rng);

    const LangevinGillespie::LGParams p = *params;
    const float drift_factor = -p.kappa / p.gammaB;
    const float diffusion_factor = sqrtf(2.0f * p.kBT / p.gammaB);
    const float sqrt_dt = sqrtf(p.dt);
    const float cycle_angle = 2.0f * CUDART_PI_F / 3.0f;

    for (int s = 0; s < sim_per_thread; ++s) {
        int sim_idx = start_sim + s;
        if (sim_idx >= nSim) break;

        int current_state = (int)p.initial_state;
        float current_angle = p.theta_0;
        int cycle_count = 0;

        // write initial values
        bead_positions_out[sim_idx * p.steps] = current_angle;
        states_out[sim_idx * p.steps] = current_state;
        target_theta_out[sim_idx * p.steps] = p.theta_states[current_state];

        for (int i = 1; i < (int)p.steps; ++i) {
            // compute outgoing rates to the three *other* states
            // j1 = (current_state + 1) % 4, j2 = +2, j3 = +3
            int j1 = (current_state + 1) & 3; // faster mod 4
            int j2 = (current_state + 2) & 3;
            int j3 = (current_state + 3) & 3;

            float rate1 = p.transition_matrix[current_state * 4 + j1];
            float rate2 = p.transition_matrix[current_state * 4 + j2];
            float rate3 = p.transition_matrix[current_state * 4 + j3];
            float total_rate = rate1 + rate2 + rate3;

            int new_state = current_state;

            if (total_rate > 0.0f) {
                // probability a jump occurs in this dt
                float p_react = 1.0f - __expf(-total_rate * p.dt);
                float u1 = curand_uniform(&rng); // in (0,1]
                if (u1 < p_react) {
                    // choose which outgoing reaction, proportional to rates
                    float u2 = curand_uniform(&rng); // in (0,1]
                    float r = u2 * total_rate;
                    if (r <= rate1) new_state = j1;
                    else if (r <= rate1 + rate2) new_state = j2;
                    else new_state = j3;

                    // cycle counting (assuming states 0..3 arranged around circle)
                    if (current_state == 3 && new_state == 0) cycle_count += 1;
                    else if (current_state == 0 && new_state == 3) cycle_count -= 1;
                }
            }

            float new_theta = p.theta_states[new_state] + cycle_count * cycle_angle;

            // Langevin update (Euler-Maruyama style here — matches your GPU code)
            float eta = curand_normal(&rng);
            float drift = drift_factor * (current_angle - new_theta);
            current_angle += p.dt * drift + diffusion_factor * sqrt_dt * eta;

            bead_positions_out[sim_idx * p.steps + i] = current_angle;
            states_out[sim_idx * p.steps + i] = new_state;
            target_theta_out[sim_idx * p.steps + i] = new_theta;

            current_state = new_state;
        }
    }
}

