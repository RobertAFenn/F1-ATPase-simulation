import sys
from pathlib import Path
import math
import numpy as np
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bin.f1sim import LangevinGillespie
from src.utils.compute_transition_matrix import compute_transition_matrix

BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
PURPLE = "\033[95m"
RESET = "\033[0m"


def create_valid_LG() -> LangevinGillespie:
    LG = LangevinGillespie()
    LG.steps = 2000
    LG.dt = 1e-6
    LG.method = "heun"
    LG.kappa = 56
    LG.kBT = 4.14
    LG.gammaB = LG.computeGammaB(a=20, r=19, eta=1e-9)
    LG.theta_states = np.array([3, 36, 72, 116]) * math.pi / 180
    LG.initial_state = 0
    LG.transition_matrix = compute_transition_matrix(LG)
    return LG


def test_simulate_multithreaded_cuda(
    LG: LangevinGillespie, nSims: int
) -> tuple[bool, str]:
    try:
        a, b, c = LG.simulate_multithreaded_cuda(nSims)
        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert isinstance(c, np.ndarray)

        assert all(isinstance(arr, np.ndarray) for arr in a)
        assert all(isinstance(arr, np.ndarray) for arr in b)
        assert all(isinstance(arr, np.ndarray) for arr in c)

        # Check individual 2D matrix values
        assert np.issubdtype(a.dtype, np.floating)
        assert np.issubdtype(b.dtype, np.integer)
        assert np.issubdtype(c.dtype, np.floating)

        return True, "Simulation ran successfully"

    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"


def run_boundary_tests(LG, param_name, test_values, nSims):
    param_results = []
    for value, should_pass in test_values:
        setattr(LG, param_name, value)
        actual_pass, msg = test_simulate_multithreaded_cuda(LG, nSims)

        passed = (actual_pass and should_pass) or (not actual_pass and not should_pass)
        param_results.append((value, should_pass, passed, msg))
        status_color = GREEN if passed else RED
        result_text = "PASS" if passed else "FAIL"
        expectation = "expect success" if should_pass else "expect failure"
        print(
            f"{PURPLE}Testing {BOLD}{param_name}{RESET} = {value} ({expectation}) → {status_color}{result_text}{RESET}"
        )
        if not passed:
            print(f"{RED}❌ Simulation outcome mismatch: {msg}{RESET}")
        # If we expected a failure and the simulation actually failed, print the python error too
        if not should_pass and not actual_pass:
            print(f"{RED}🆗 {msg}{RESET}")
    print(f"{GREEN}🥳 Completed boundary tests for {BOLD}{param_name}{RESET}{RESET}")
    print(f"{PURPLE}Re-initializing a valid LG for next parameter...{RESET}\n")
    return create_valid_LG(), param_results


def print_summary(all_results):
    print(f"\n{BOLD}【=◈ Summary LG Params Only =◈】{RESET}")
    for param, results in all_results.items():
        param_status = (
            GREEN + "✅" + RESET
            if all(passed for _, _should_pass, passed, _msg in results)
            else RED + "❌" + RESET
        )
        print(f"{BOLD}{param}{RESET}: {param_status}")
        for value, should_pass, passed, msg in results:
            status = GREEN + "PASS" + RESET if passed else RED + "FAIL" + RESET
            print(f"   {value}: {status}")
            # Print message if the test failed or if we expected a failure (to show the python error)
            if (not passed) or (not should_pass):
                print(f"        🆗 {RED}{msg}{RESET}")


def print_test_result(param, value, passed, msg=None, expect="success"):
    expected_pass = True if expect == "success" else False
    correct = passed == expected_pass
    status_color = GREEN if correct else RED
    result_text = "PASS" if correct else "FAIL"

    print(
        f"{PURPLE}Testing {BOLD}{param}{RESET} = {value} (expect {expect}) → {status_color}{result_text}{RESET}"
    )
    if msg:
        print(f"🆗 {RED}{msg}{RESET}")


# Test Start
nSims = 1000

LG = create_valid_LG()
passed, msg = test_simulate_multithreaded_cuda(LG, nSims)
print_test_result("Valid LG", nSims, passed, msg, expect="success")

LG = LangevinGillespie()
passed, msg = test_simulate_multithreaded_cuda(LG, nSims)
print_test_result("Unfilled LG", nSims, passed, msg, expect="failure")

# Boundary tests
print("\nStarting the boundary input test(s)")

nSims = 0
LG = create_valid_LG()
passed, msg = test_simulate_multithreaded_cuda(LG, nSims)
print_test_result("nSims", nSims, passed, msg, expect="failure")
nSims = 1000
print()

all_results = {}
LG = create_valid_LG()


LG, all_results["steps"] = run_boundary_tests(
    LG,
    "steps",
    [(0, False), (1, True), (10_000, True), (100_000, True)],
    nSims,
)
LG, all_results["dt"] = run_boundary_tests(
    LG, "dt", [(0, False), (0.0001, True), (0.001, True), (0.01, True)], nSims
)
LG, all_results["method"] = run_boundary_tests(
    LG,
    "method",
    [
        ("You should listen to Porter Robinson", False),
        ("heun", True),
        ("euler", True),
        ("probabilistic", True),
    ],
    nSims,
)
LG, all_results["kappa"] = run_boundary_tests(
    LG, "kappa", [(-1, False), (0, True)], nSims
)
LG, all_results["kBT"] = run_boundary_tests(LG, "kBT", [(-1, False), (0, True)], nSims)
LG, all_results["gammaB"] = run_boundary_tests(
    LG, "gammaB", [(-1, False), (1e-9, True)], nSims
)

# Transition matrix size tests only
transition_matrices = [
    ([], False),
    ([[0.25] * 4] * 3, False),
    ([[0.25] * 3] * 4, False),
    ([[0.25] * 4] * 5, False),
    ([[0.25] * 4] * 4, True),
]
LG, all_results["transition_matrix"] = run_boundary_tests(
    LG, "transition_matrix", transition_matrices, nSims
)

LG, all_results["initial_state"] = run_boundary_tests(
    LG, "initial_state", [(0, True), (1, True), (2, True), (3, True)], nSims
)
LG, all_results["theta_states"] = run_boundary_tests(
    LG,
    "theta_states",
    [
        ([], False),
        ([0, 0, 0], False),
        ([0, 0, 0, 0, 0], False),
        ([0, 0, 0, 0], True),
        ([math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2], True),
    ],
    nSims,
)

print_summary(all_results)
