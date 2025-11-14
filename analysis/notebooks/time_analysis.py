#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('cd', '.. ..')
import math
import numpy as np
import time
import torch
from archive.python.LangevinGillespie import LangevinGillespie
from src.utils.compute_transition_matrix import compute_transition_matrix
from bin.f1sim import LangevinGillespie as LangevinGillespie_PybindWrap


# In[9]:


get_ipython().system('fastfetch --logo none --structure os:kernel:cpu:gpu')


# In[3]:


def initialize_simulation_params(LG):
    LG.steps = 2000
    LG.dt = 1e-6
    LG.method = "heun"

    # Mechanical / Thermal Setup
    LG.kappa = 56
    LG.kBT = 4.14
    LG.gammaB = LG.computeGammaB(a=20, r=19, eta=1e-9)

    # Multi State Setup
    LG.theta_states = np.array([3, 36, 72, 116]) * math.pi / 180  # Deg → Rad
    LG.initial_state = 0  # Starting state

    # Transition rate matrix
    LG.transition_matrix = compute_transition_matrix(LG)


# In[4]:


# Initialize simulation wrapper
LG_PybindWrap = LangevinGillespie_PybindWrap()
initialize_simulation_params(LG_PybindWrap)

N_SIMS = 10000

BYTES_PER_FLOAT = 4
BYTES_PER_INT = 4
TOTAL_FLOATS_PER_SIM = 3 * LG_PybindWrap.steps  # bead_positions, target_theta, etc.
TOTAL_INTS_PER_SIM = LG_PybindWrap.steps  # states
BYTES_PER_SIM = (
    TOTAL_FLOATS_PER_SIM * BYTES_PER_FLOAT + TOTAL_INTS_PER_SIM * BYTES_PER_INT
)

# --- Dynamic GPU memory check ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    total_mem = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    free_mem = total_mem - allocated - reserved

    # Leave some room for other GPU usage
    MAX_MEMORY_BYTES = int(free_mem * 0.5)
else:
    # fallback for CPU only
    MAX_MEMORY_BYTES = 8 * 1024**3  # GB

# Compute batch size
BATCH_SIZE = max(1, min(N_SIMS, MAX_MEMORY_BYTES // BYTES_PER_SIM))
total_batches = math.ceil(N_SIMS / BATCH_SIZE)

print(f"Batch size: {BATCH_SIZE}, total batches: {total_batches}")


# ##### Note: If you use multi-threading, avoid using swap memory, instead employ batching

# In[5]:


start_time = time.time()
for batch_start in range(0, N_SIMS, BATCH_SIZE):
    n_batch = min(BATCH_SIZE, N_SIMS - batch_start)
    batch_num = batch_start // BATCH_SIZE + 1

    print(f"\rRunning batch {batch_num}/{total_batches}: {n_batch} simulations", end="", flush=True)

    # Run CUDA kernel
    beads, states, thetas = LG_PybindWrap.simulate_multithreaded_cuda(n_batch)

print(f"\nTotal time: {time.time() - start_time:.2f} s")


# In[6]:


start_time = time.time()
for batch_start in range(0, N_SIMS, BATCH_SIZE):
    n_batch = min(BATCH_SIZE, N_SIMS - batch_start)
    batch_num = batch_start // BATCH_SIZE + 1

    print(f"\rRunning batch {batch_num}/{total_batches}: {n_batch} simulations", end="", flush=True)

    beads, states, thetas = LG_PybindWrap.simulate_multithreaded(n_batch, 32)

print(f"\nTotal time: {time.time() - start_time:.2f} s")


# In[7]:


bead_store = []
states_store = []
thetas_store = []
start_time = time.time()
for batch_start in range(0, N_SIMS, BATCH_SIZE):
    n_batch = min(BATCH_SIZE, N_SIMS - batch_start)
    batch_num = batch_start // BATCH_SIZE + 1

    print(f"\rRunning batch {batch_num}/{total_batches}: {n_batch} simulations", end="", flush=True)

    for i in range(0, n_batch):
        beads, states, thetas = LG_PybindWrap.simulate()
        bead_store.append(beads)
        states_store.append(states)
        thetas_store.append(thetas)

print(f"\nTotal time: {time.time() - start_time:.2f} s")


# In[8]:


# print(f"The C++ wrap was {time2_total / time1_total:.2f} times faster than Python!")

