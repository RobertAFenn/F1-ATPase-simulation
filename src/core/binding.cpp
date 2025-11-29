// src/core/bindings.cpp
#include "../include/LangevinGillespie.h"

// -=-=-=-=-=-=-=-=-= STD lib -=-=-=-=-=-=-=-=-=
#include <optional>       // std::optional
#include <random>         // std::mt19937, distributions
#include <chrono>         // std::chrono::steady_clock
#include <cstdint>        // fixed-width integer types

// -=-=-=-=-=-=-=-=-= PYBIND11 -=-=-=-=-=-=-=-=-=
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// splitmix64 - small, fast 64-bit mix function to expand / scramble seed material
static uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// produce a robust 64-bit seed using random_device + time fallback
static unsigned long long make_random_seed64() {
    std::random_device rd;
    // gather some 32-bit values (rd may or may not be non-deterministic)
    uint64_t a = static_cast<uint64_t>(rd()) & 0xffffffffULL;
    uint64_t b = static_cast<uint64_t>(rd()) & 0xffffffffULL;

    // mix with high-res time to reduce chance of collisions on poor rd implementations
    uint64_t t = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uint64_t state = (a << 32) ^ b ^ t;

    // produce two splitmix64 outputs and combine
    uint64_t s1 = splitmix64(state);
    uint64_t s2 = splitmix64(state);

    return (s1 << 32) ^ (s2 & 0xffffffffULL);
}

std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return result;
}

PYBIND11_MODULE(f1sim, m) {
    m.doc() = "Pybind11 wrapper exposing the LangevinGillespie class with GPU support";

    py::class_<LangevinGillespie>(m, "LangevinGillespie")
        .def(py::init<>())
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

        // -=-=-=-=-=-=-=-=-= CPU -=-=-=-=-=-=-=-=-=
        .def("computeGammaB",
            &LangevinGillespie::computeGammaB,
            py::arg("a") = 0,
            py::arg("r") = 0,
            py::arg("eta") = 0,
            "Compute rotational friction coefficient of the bead.")

        .def("simulate",
            &LangevinGillespie::simulate,
            py::arg("seed") = std::nullopt,
            "Run a Langevin simulation on CPU.\n"
            "Returns bead_positions, states, target_thetas")

        .def("simulate_multithreaded",
            &LangevinGillespie::simulate_multithreaded,
            py::arg("nSim"),
            py::arg("num_threads"),
            py::arg("seed") = std::nullopt,
            "Run multiple Langevin simulations in parallel on CPU.\n"
            "Returns bead_positions, states, target_thetas")

        // -=-=-=-=-=-=-=-=-= GPU (Nvidia) -=-=-=-=-=-=-=-=-=
        .def("simulate_multithreaded_cuda",
            [](LangevinGillespie& self,
                int nSim,
                std::optional<unsigned long long> base_seed)
            {
                unsigned long long seed_val;
                if (!base_seed.has_value()) seed_val = make_random_seed64();
                else seed_val = base_seed.value();
                return self.simulate_multithreaded_cuda(nSim, seed_val);
            },
            py::arg("nSim"),
            py::arg("seed") = std::nullopt,
            "Run multiple Langevin simulations on GPU via CUDA.\n"
            "If base_seed is omitted (None), a random 64-bit seed is generated at the binding level.");

}
