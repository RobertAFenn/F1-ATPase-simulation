# F1-ATPase Simulation

A Python project that simulates 1D Langevin dynamics using stochastic integration
methods such as Heun, Euler–Maruyama, and a probabilistic approach.
It models rotational dynamics of a particle under the influence of elastic
restoring forces and thermal fluctuations.

The project includes CPU and GPU implementations for efficient computation,
along with analysis scripts and Juypter notebooks in which explore simulation
results

---

## Overview

This project can be used to study:

- Rotational motion of a particle or molecular motor (e.g., F1-ATPase)
- Effects of thermal fluctuations on angular dynamics
- Performance of different stochastic integration methods (Heun, Euler, Probabilistic)

The simulation captures how a particle evolves toward an equilibrium position
under both deterministic and stochastic forces.

---

## Project Structure

```
F1-ATPase-simulation/
├── analysis/ # Jupyter notebooks and scripts for analyzing simulation results
│ ├── notebooks/ # Experiment notebooks
│ │   └── time_analysis.ipynb # Compare different LangevinGillespie computing methods
│ └── results/ # Generated figures, CSVs, and other outputs
├── bin/ # Compiled binaries and Python extensions
│   └── f1sim.cpython-311-x86_64-linux-gnu.so
├── docs/ # Project documentation, diagrams, flowcharts, roadmap
├── src/ # Source code
│ ├── core/ # Core simulation logic
│ │ ├── cpp/ # CPU-based LangevinGillespie implementation
│ │ │   └── LangevinGillespie.cpp
│ │ ├── cuda/ # GPU-based LangevinGillespie implementation
│ │ │   └── LangevinGillespie.cu
│ │ ├── include/ # Header files
│ │ │   └── LangevinGillespie.h
│ │ └── bindings/ # C++ → Python interface
│ │     └── binding.cpp
│ └── utils/ # Helper scripts and functions
│     └── compute_transition_matrix.py # Compute transition matrix for F1-ATPase
├── main.ipynb # Notebook for running simulations and experiments
├── requirements.txt # Python dependencies
└── setup.py # Python package setup
```

---

## Setup

Clone the repository:

    git clone https://github.com/RobertAFenn/F1-ATPase-simulation.git
    cd F1-ATPase-simulation

Or, if you prefer, you can download just the bin file:

    bin/f1sim.cpython-311-x86_64-linux-gnu.so
> **Note**: The binary is built for Python 3.11 on linux x86_64.
> Using it on other systems may require compiling from the source.
> The `setup.py` file is provided for this purpose as it allows you to
> build the Python extension on your own system. 

The main.ipynb notebook provides examples on how to use the class.
All public methods from C++ to Python can be found documented in the binding file.
    
    src/core/binding.cpp

In depth method information can be found in the header file.

    src/core/include/LangevinGillespie.h


(Optional) Create a virtual environment:

    python -m venv myVenv
    source myVenv/bin/activate   # macOS/Linux
    myVenv\Scripts\activate      # Windows
    pip install -r requirements.txt

---

## Roadmaps

### Project Flowchart

![Flowchart](docs/flowchart.png)

### Project Roadmap

![Project Roadmap](docs/project_roadmap.png)

## Usage Example

```python
## Usage Example

import math
import numpy as np
from bin.f1sim import LangevinGillespie
from src.utils.compute_transition_matrix import compute_transition_matrix

# -=-=-=-=-=-=-=-=-=-=-=-
# Simulation Setup
# -=-=-=-=-=-=-=-=-=-=-=-
LG = LangevinGillespie()
LG.steps = 2000
LG.dt = 1e-6
LG.method = "heun"

# -=-=-=-=-=-=-=-=-=-=-=-
# Mechanical / Thermal Setup
# -=-=-=-=-=-=-=-=-=-=-=-
LG.kappa = 56
LG.kBT = 4.14
LG.gammaB = LG.computeGammaB(a=20, r=19, eta=1e-9)

# -=-=-=-=-=-=-=-=-=-=-=-
# Multi-State Setup
# -=-=-=-=-=-=-=-=-=-=-=-
LG.theta_states = np.array([3, 36, 72, 116]) * math.pi / 180  # degrees → radians
LG.initial_state = 0  # starting state [0-3]

# -=-=-=-=-=-=-=-=-=-=-=-
# Transition Rate Matrix
# -=-=-=-=-=-=-=-=-=-=-=-
LG.transition_matrix = compute_transition_matrix(LG)

# -=-=-=-=-=-=-=-=-=-=-=-
# 1) Single-threaded simulation
# -=-=-=-=-=-=-=-=-=-=-=-
angles, states, thetas = LG.simulate(seed=42)
# shapes:
# angles.shape -> (steps)
# states.shape -> (steps)
# thetas.shape -> (steps)

# -=-=-=-=-=-=-=-=-=-=-=-
# 2) Multi-threaded CPU simulation
# -=-=-=-=-=-=-=-=-=-=-=-
nSim = 10
num_threads = 4
angles_all, states_all, thetas_all = LG.simulate_multithreaded(nSim=nSim, num_threads=num_threads, seed=42)
# shapes:
# angles_all.shape -> (nSim, steps)
# states_all.shape -> (nSim, steps)
# thetas_all.shape -> (nSim, steps)

# -=-=-=-=-=-=-=-=-=-=-=-
# 3) Multi-threaded GPU (CUDA) simulation
# -=-=-=-=-=-=-=-=-=-=-=-
angles_all_gpu, states_all_gpu, thetas_all_gpu = LG.simulate_multithreaded_cuda(nSim=nSim, seed=42)
# shapes:
# angles_all_gpu.shape -> (nSim, steps), dtype=float32
# states_all_gpu.shape -> (nSim, steps), dtype=int32
# thetas_all_gpu.shape -> (nSim, steps), dtype=float32
```

## References

- ["Method to extract multiple states in F1-ATPase rotation experiments from jump distributions"](https://www.pnas.org/doi/10.1073/pnas.1915314116)
- [F1-ATPase_torque_Langevin_simulation](https://github.com/canhochoi/F1-ATPase_torque_Langevin_simulation)
