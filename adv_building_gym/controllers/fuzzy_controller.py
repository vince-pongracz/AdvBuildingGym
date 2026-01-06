# File: adv_building_gym/controllers/fuzzy_controller.py
import numpy as np


# FuzzyController
class FuzzyController:
    """
    A fuzzy controller for a dual-mode heat pump system with optional Gaussian
    membership functions and optional debug prints for traceability.
    Dual-Mode Behavior:
    -------------------
    - Positive fuzzy output => heating
    - Negative fuzzy output => cooling

    Membership Functions:
    ---------------------
    Users can select triangular or Gaussian membership functions. The 'predict'
    method calculates a weighted centroid for defuzzification.

    """

    def __init__(self, debug=False, use_gaussian=False, sigma=2.0, fine_tuning=False):
        """
        Initialize fuzzy controller parameters.

        Parameters:
            debug (bool): If True, prints intermediate membership and output
                          values for debugging.
            use_gaussian (bool): If True, uses Gaussian membership; otherwise,
                                 uses triangular membership functions.
            sigma (float): Std for Gaussian membership functions.
            fine_tuning (bool): If True, uses a refined set of parameters for
                                the fuzzy sets.
        """
        self.debug = debug
        self.use_gaussian = use_gaussian
        self.sigma = sigma
        self.fine_tuning = fine_tuning
        # Scale factors for fine-tuning membership shapes and outputs
        output_factor = 1.5
        b_factor = 0.75

        if fine_tuning:
            # Fine-tuned fuzzy set parameters: (a, b, c, output)
            self.fuzzy_params = {
                "VC": (-2.5, b_factor * -2.0, -1.5, output_factor * 1.0),
                "C": (-1.5, b_factor * -1.0, -0.5, output_factor * 0.5),
                "I": (-0.5, b_factor * 0.0, 0.5, output_factor * 0.0),
                "H": (0.5, b_factor * 1.0, 2.5, output_factor * -0.5),
                "VH": (2.5, b_factor * 2.0, 7.5, output_factor * -1.0),
            }
            self.gaussian_error_scale = self.sigma
        else:
            # Paper Plot Settings
            self.fuzzy_params = {
                "VC": (-6.0, -4.0, -2.0, 1.0),
                "C": (-4.0, -2.0, 0.0, 0.5),
                "I": (-2.0, 0.0, 2.0, 0.0),
                "H": (0.0, 2.0, 4.0, -0.5),
                "VH": (2.0, 4.0, 6.0, -1.0),
            }
            self.gaussian_error_scale = self.sigma

    def triangle(self, e, a, b, c):
        """Triangular membership function μ(e; a, b, c)."""
        if e <= a or e >= c:
            return 0.0
        elif e == b:
            return 1.0
        elif e < b:
            return (e - a) / (b - a)
        else:
            return (c - e) / (c - b)

    def membership(self, e, a, b, c):
        """
        Compute the membership degree using either a triangular or Gaussian.

        Parameters:
            e (float): The error or deviation from setpoint.
            a, b, c (floats): Shape-defining parameters for the
                              membership function.

        Returns:
            float: Degree of membership in [0, 1].
        """
        if self.use_gaussian:
            effective_error = e * self.gaussian_error_scale
            return np.exp(-0.5 * ((effective_error - b) / self.sigma) ** 2)
        else:
            return self.triangle(e, a, b, c)

    def predict(self, obs, deterministic=True):
        """
        Given an observation (usually [delta_T, ...]),
        compute the fuzzy control action.

        Parameters:
            obs (np.array): Observation, where obs[0] is typically the
                            temperature deviation from setpoint.
            deterministic (bool): Not used here, but included for API
                                  compatibility.

        Returns:
            (action, None): The first element is the control action in [-1, 1],
                            consistent with the environment’s action space.
                            Here, a negative sign is applied to invert fuzzy
                            sign => environment’s sign convention.
        """
        error = obs[0]
        # Clip extreme errors to avoid undefined membership computations
        error_max = 15
        error_min = -5

        if error > error_max:
            if self.debug:
                print(f"[Fuzzy] Extreme error {error:.2f} > {error_max}, max cooling.")
            return np.array([self.fuzzy_params["VH"][3]]), None
        elif error < error_min:
            if self.debug:
                print(f"[Fuzzy] Extreme error {error:.2f} < {error_min}, max heating.")
            return np.array([self.fuzzy_params["VC"][3]]), None

        numerator = 0.0
        denominator = 0.0
        # Compute weighted average (centroid) of outputs
        for key, (a, b, c, output) in self.fuzzy_params.items():
            mu = self.membership(error, a, b, c)
            numerator += mu * output
            denominator += mu
        # Default to 0 if denominator is extremely small
        action = numerator / denominator if denominator != 0 else 0.0

        if self.debug:
            print(f"[Fuzzy] Error = {error:.2f}")
            for key, (a, b, c, output) in self.fuzzy_params.items():
                mu = self.membership(error, a, b, c)
                print(f" {key}: membership={mu:.2f}, output={output}")
            print(f"  => action = {action:.2f}")
        # Return the action as negative because we interpret
        # positive membership as "cooling" in this example:
        return -np.array([action]), None
