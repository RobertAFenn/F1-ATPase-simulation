import random
import math

"""
Notes: 
It seems that whole thing about theta_i is that it changes in multi state
This means of course you don't need to worry about it for now
"""


class LangevinGillespie:
    def __init__(self):
        # Simulation Parameters
        self.steps = None  # Number of time steps : int
        self.dt = None  # Time step size : float
        self.method = None  # Integration method: "heun", "euler", "probabilistic"

        # Physical System Parameters
        self.theta_0 = None  # Initial Position : degrees (int / float)
        self.theta_i = None  # Position the system settles in : degrees (float)
        self.gammaB = None  # Rotational friction: float
        self.kappa = None  # Elastic constant: float
        self.kBT = None  # Thermal energy: float

    def simulate(self, rng=None) -> list[float]:
        """
        Run a Langevin-Gillespie simulation.

        Parameters
        ----------
        rng : int, optional
            Random rng for reproducibility. Default is None.

        Returns
        -------
        List[float]
            Angular positions (angle in degrees) over time.
        """

        verify_attributes(self)  # Error Checking

        method = self.method.lower()  # Make method lowercase so string matching works

        rng = random.Random(rng) if rng is not None else random  # Local reproducibility

        angles = [0] * self.steps
        target_angle = self.theta_i

        angles[0] = self.theta_0
        for i in range(1, self.steps):
            match method:  # Every case should be lowercase
                case "heun":
                    angles[i] = heun_1d(angles[i - 1], target_angle, self.dt, self.gammaB, self.kappa, self.kBT, rng=rng)

                case "euler":
                    angles[i] = euler_maruyama(angles[i - 1], target_angle, self.dt, self.gammaB, self.kappa, self.kBT, rng=rng)

                case "probabilistic":
                    angles[i] = compute_probabilistic(angles[i - 1], target_angle, self.dt, self.gammaB, self.kappa, self.kBT, rng=rng)

                case _:
                    raise ValueError("ERROR: Method must be a string and defined as one of the following (heun, euler, probabilistic)")

        return angles

    @staticmethod
    def computeGammaB(a: float, r: float, eta: float) -> float:
        """
        Compute rotational friction coefficient GammaB for a spherical particle.

        Parameters
        ----------
        a : float
            Radius of the sphere (nm).
        r : float
            Distance from rotational axis to sphere center (nm).
        eta : float
            Dynamic viscosity of the fluid (pN.s/nm^2).

        Returns
        -------
        float
            Rotational friction coefficient (gammaB).
        """
        return 8 * math.pi * eta * a**3 + 6 * math.pi * eta * a * r**2

    def initializeTheta(self, angle: float, rng=random) -> float:
        """
        Initialize angle with a random thermal fluctuation.

        Parameters
        ----------
        angle : float
            Base angle (degrees).

        Returns
        -------
        float
            Angle after adding thermal fluctuation.
        """
        return math.sqrt(self.kBT / (10 * self.kappa)) * rng.gauss(0, 1) + angle


# Helper Functions (Ordered by when they appear)
def verify_attributes(LG: LangevinGillespie) -> None:
    """
    Ensure all required attributes of LangevinGillespie are set.

    Parameters
    ----------
    LG : LangevinGillespie
        Instance to verify.

    Raises
    ------
    ValueError
        If any required attribute is None.
    """
    required_attributes = ["steps", "dt", "theta_0", "theta_i", "gammaB", "kappa", "kBT", "method"]
    missing_attributes = ""
    for attr in required_attributes:
        if getattr(LG, attr) is None:
            missing_attributes += "[" + (attr) + "]"
    if missing_attributes != "":
        raise ValueError("The following attribute(s) are Null and must be defined (" + missing_attributes + ")")


def drift(theta_0: float, gammaB: float, kappa: float, theta_i: float) -> float:
    """Compute deterministic drift term for angular motion."""
    return 1 / gammaB * (-kappa * (theta_0 - theta_i))


def diffusion(gammaB: float, kBT: float) -> float:
    """Compute stochastic diffusion term for angular motion."""
    return math.sqrt(2 * kBT / gammaB)


def heun_1d(theta_0: float, theta_i: float, dt: float, gammaB: float, kappa: float, kBT: float, rng=random) -> float:
    """Heun (predictor-corrector) method step."""
    driftX = drift(theta_0, gammaB, kappa, theta_i)
    diffusionX = diffusion(gammaB, kBT)
    eta = rng.gauss(0, 1)  # standard normal

    y_predict = theta_0 + dt * driftX + math.sqrt(dt) * diffusionX * eta  # Predictor step
    drift_predict = drift(y_predict, gammaB, kappa, theta_i)  # Corrector step

    theta_0_next = theta_0 + (dt / 2) * (driftX + drift_predict) + math.sqrt(dt) * diffusionX * eta

    return theta_0_next


def euler_maruyama(theta_0: float, theta_i: float, dt: float, gammaB: float, kappa: float, kBT: float, rng=random) -> float:
    """Euler-Maruyama stochastic integration step."""
    return theta_0 + dt * (1 / gammaB) * (-kappa * (theta_0 - theta_i)) + math.sqrt(2 * kBT / gammaB * dt) * rng.gauss(0, 1)


def compute_probabilistic(theta_0: float, theta_i: float, dt: float, gammaB: float, kappa: float, kBT: float, rng=random) -> float:
    """Simplified probabilistic integration step (same as Euler-Maruyama)."""
    return theta_0 + dt * (1 / gammaB) * (-kappa * (theta_0 - theta_i)) + math.sqrt(2 * kBT / gammaB * dt) * rng.gauss(0, 1)
