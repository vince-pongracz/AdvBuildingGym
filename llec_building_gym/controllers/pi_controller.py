# File: llec_building_gym/controllers/pi_controller.py
import numpy as np


# PIController
class PIController:
    """
    Proportional-Integral (PI) Controller for a dual-mode heat pump.

    Control Law:
    ------------
       u(t) = Kp * e(t) + Ki * ∫ e(t) dt

    where e(t) is the temperature error (T_in - T_set).
    """

    def __init__(self, Kp=0.15, Ki=0.0002, dt=300):
        """
        Initialize PI controller gains and state.

        Parameters:
            Kp (float): Proportional gain.  (0.15)
            Ki (float): Integral gain.      (0.0002)
            dt (float): Sampling time step for integration (s). (300)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.dt = dt
        self.integral = 0.0
        self.horizon = 1

    def predict(self, obs, deterministic=True):
        """
        Compute the PI control output based on the current observation.

        Parameters:
            obs (np.array): The observation, typically [temperature_deviation].
            deterministic (bool): Ignored in this deterministic controller.

        Returns:
            (action, None): A single-element numpy array
                            with the clipped action.
        """
        error = obs[0]
        self.integral += error * self.dt
        action = self.Kp * error + self.Ki * self.integral
        action = np.clip(action, -1.0, 1.0)
        return np.array([action]), None
