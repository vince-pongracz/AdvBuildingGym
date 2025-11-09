# File: llec_building_gym/controllers/pid_controller.py
import numpy as np


# PIDController
class PIDController:
    """
    Proportional-Integral-Derivative (PID) Controller for a dual-mode
    heat pump.

    Control Law:
    ------------
       u(t) = Kp*e(t) + Ki*∫ e(t) dt + Kd * d/dt[e(t)]

    where e(t) is the temperature error (T_in - T_set).
    """

    def __init__(self, Kp=0.15, Ki=0.0002, Kd=0.01, dt=300):
        """
        Initialize PID controller gains and internal states.

        Parameters:
            Kp (float): Proportional gain.  (0.15)
            Ki (float): Integral gain.      (0.0002)
            Kd (float): Derivative gain.    (0.01)
            dt (float): Sampling time step (s). (300)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = None

    def predict(self, obs, deterministic=True):
        """
        Compute the PID control action for the current observation.

        Parameters:
            obs (np.array): The observation, typically [temperature_deviation].
            deterministic (bool): Unused in this deterministic controller.

        Returns:
            (action, None): A single-element numpy array containing the
                            clipped action in [-1, 1].
        """
        error = obs[0]

        # Integrate error over time
        self.integral += error * self.dt

        # Compute derivative using a finite difference approach
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / self.dt

        self.prev_error = error

        action = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        action = np.clip(action, -1.0, 1.0)
        return np.array([action]), None
