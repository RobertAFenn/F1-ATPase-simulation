# F1-ATPase Simulation

A Python project that simulates 1D Langevin dynamics using stochastic integration
methods such as Heun, Euler–Maruyama, and a probabilistic approach.
It models rotational dynamics of a particle under the influence of elastic
restoring forces and thermal fluctuations.

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
├── docs/ # Flowcharts, project roadmap, diagrams
├── src/ # Source code
│ ├── core/ # Main simulation classes (LangevinGillespie)
│ ├── utils/ # Helper functions (math, initialization, plotting)
│ └── config/ # Optional: constants and default parameters
├── analysis/ # Jupyter notebooks and analysis scripts
├── results/ # Generated figures and data (optional)
├── main.ipynb # Notebook for running simulations and experiments
└── requirements.txt # Python dependencies
```

---

## Setup

Clone the repository:

    git clone https://github.com/RobertAFenn/F1-ATPase-simulation.git
    cd F1-ATPase-simulation

Or, if you prefer, you can download just the simulation class:

    src/core/LangevinGillespie.py

The main.ipynb notebook provides examples on how to use the class, and
all methods are documented inside the Python file.

(Optional) Create a virtual environment:

    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows
    pip install -r requirements.txt

---

## Roadmaps

### Project Flowchart

![Flowchart](docs/flowchart.png)

### Project Roadmap

![Project Roadmap](docs/project_roadmap.png)

---

## Usage Example

    from src.core.LangevinGillespie import LangevinGillespie
    import matplotlib.pyplot as plt

    r = 19  # nm - distance from the rotational axis to the center of the sphere
    a = 20  # nm - radius of a sphere (See https://en.wikipedia.org/wiki/Stokes_flow)
    eta = 1e-9  # pN.s/nm^2

    LG = LangevinGillespie()
    LG.steps = 1000
    LG.dt = 1e-6
    LG.kappa = 56
    LG.kBT = 4.14
    LG.gammaB = LG.computeGammaB(a, r, eta)
    LG.method = "heun"

    SIM_ANGLE = 0
    TARGET_ANGLE = SIM_ANGLE
    LG.theta_0 = LG.initializeTheta(SIM_ANGLE)
    LG.theta_i = TARGET_ANGLE

    simulation_data = LG.simulate()

This will return a list of angular positions (θ) over time for the specified parameters.

---

## References

- ["Method to extract multiple states in F1-ATPase rotation experiments from jump distributions"](https://www.pnas.org/doi/10.1073/pnas.1915314116)
- [F1-ATPase_torque_Langevin_simulation](https://github.com/canhochoi/F1-ATPase_torque_Langevin_simulation)
