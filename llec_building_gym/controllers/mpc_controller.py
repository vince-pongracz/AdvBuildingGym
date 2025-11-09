# File: llec_building_gym/controllers/mpc_controller.py
import numpy as np
import pyomo.environ as pyo


# MPCController
class MPCController:
    """
    A simplified Model Predictive Controller (MPC) that can optimize either
    temperature objectives alone or a combination of temperature and economic
    objectives, depending on the reward_mode.
    MPC controller with optional objective:
        - 'temperature' or 'combined' (incl. economic costs).
    ----------------------------------------
    This controller is based on a simplified building model and optimizes over
    a short horizon. This class implements a minimal MPC scheme over a short time
    horizon. In `predict`, a pyomo model is constructed, solved and the resulting
    first control value (action) is returned.
    """

    def __init__(
        self,
        dt=300,
        horizon=3,
        building_params=None,
        reward_mode="temperature",
        temperature_weight=1.0,
        economic_weight=1.0,
    ):
        """
        Initialize the MPC parameters and the optimization preferences.
        Args:
            dt (int, optional): Discretization timestep in seconds. Defaults to 300.
            horizon (int, optional): Number of prediction steps in the optimization.
                Defaults to 3.
            building_params (dict, optional): Contains building constants
                (mC, K, Q_HP_Max, T_set). If None, default values are used.
            reward_mode (str, optional): Either "temperature" or "combined".
                Determines the form of the objective function. Defaults to "temperature".
            temperature_weight (float, optional): Weight for the temperature objective
                in the combined mode. Defaults to 1.0.
            economic_weight (float, optional): Weight for the economic objective
                in the combined mode. Defaults to 1.0.
        """
        self.dt = dt
        self.horizon = horizon
        self.reward_mode = reward_mode
        self.temperature_weight = temperature_weight
        self.economic_weight = economic_weight

        # Default building parameters if none are provided
        if building_params is None:
            # Default values, if not passed
            self.building_params = {"mC": 300, "K": 20, "Q_HP_Max": 1500, "T_set": 25.0}
        else:
            self.building_params = building_params

    def predict(self, obs, deterministic=True, **kwargs):
        """
        Builds a Pyomo optimization model, solves it, and returns the first optimal action.

        Args:
            obs (np.array): Observations from the environment.
                Typically, obs[0] = T_in - T_set, so indoor temperature can be recovered.
            deterministic (bool, optional): Ignored in this simple MPC,
                but kept for interface consistency. Defaults to True.
            **kwargs:
                T_out_pred (list or np.array): Outdoor temperature forecast of length horizon+1.
                price_pred (list or np.array): Energy price forecast of length horizon.
                Additional parameters could be passed here if needed.

        Returns:
            tuple:
                - action (np.array): The first control action in the interval [-1, 1].
                - None: Placeholder for compatibility with controllers returning (action, state).
        """
        # 1) Extract relevant state variables
        # If obs[0] = (T_in - T_set) => T_in = obs[0] + T_set
        # Planning horizon in number of steps
        H = self.horizon
        T_set_list = list(
            kwargs.get(
                "T_set_pred", [self.building_params["T_set"]] * (self.horizon + 1)
            )
        )

        if len(T_set_list) < self.horizon + 1:
            last_value = (
                T_set_list[-1] if len(T_set_list) > 0 else self.building_params["T_set"]
            )
            T_set_list = list(T_set_list) + [last_value] * (
                self.horizon + 1 - len(T_set_list)
            )
        else:
            T_set_list = list(T_set_list[: self.horizon + 1])

        T_in_current = float(obs[0] + T_set_list[0])

        # Retrieve or default to dummy predictions
        T_out_list = kwargs.get("T_out_pred", [30.0] * (self.horizon + 1))
        # price_list = kwargs.get("price_pred", [0.5] * self.horizon)

        mC = self.building_params["mC"]
        K = self.building_params["K"]
        Q_HP_Max = self.building_params["Q_HP_Max"]
        dt = self.dt

        # Safely retrieve T_out_pred
        T_out_pred = kwargs.get("T_out_pred", None)
        print(f"[DEBUG] Using T_set: {T_set_list}")

        # 2) Set T_out_list
        if T_out_pred is None:
            # Use dummy values if no prediction was given
            T_out_list = [30.0] * (self.horizon + 1)
            print(f"[DEBUG] Using dummy T_out_pred: {T_out_list}")

        else:
            T_out_pred = list(T_out_pred)  # Ensure it's a list
            if len(T_out_pred) < self.horizon + 1:
                print(
                    f"[DEBUG] T_out_pred too short (got {len(T_out_pred)}), extending with last value."
                )
                last_value = T_out_pred[-1] if len(T_out_pred) > 0 else 30.0
                T_out_list = T_out_pred + [last_value] * (
                    self.horizon + 1 - len(T_out_pred)
                )
            else:
                T_out_list = T_out_pred[: self.horizon + 1]
            print(f"[DEBUG] Using provided T_out_pred: {T_out_list}")

        # 3) Define Pyomo model
        model = pyo.ConcreteModel()
        # Control and state variables
        model.t = pyo.RangeSet(0, H - 1)
        # State variables: T_in[t] => Temperatur Indoor
        model.T_in = pyo.Var(range(self.horizon + 1), domain=pyo.Reals)
        # Control variables: action[t] in [-1,1]
        model.action = pyo.Var(model.t, domain=pyo.Reals, bounds=(-1, 1))
        # Initial condition
        model.T_in[0].fix(T_in_current)
        scale = 0.001  # Consistent scaling factor if required by environment

        # 3) Define dynamic equations

        def dyn_rule(m, i):
            # first use T_out_list[i] here
            if i >= len(T_out_list):
                # Do not use T_out_list[i] at all
                return pyo.Constraint.Skip
            if i == H:
                return pyo.Constraint.Skip
            return m.T_in[i + 1] == m.T_in[i] - scale * (dt / mC) * (
                K * (m.T_in[i] - T_out_list[i]) + m.action[i] * Q_HP_Max
            )

        model.dynamics = pyo.Constraint(model.t, rule=dyn_rule)

        # 4) Define objective function
        def objective_rule(m):
            """
            Objective function that can be temperature-only or combined with
            an economic term, depending on self.reward_mode.
            """
            temp_penalty = sum((m.T_in[i] - T_set_list[i]) ** 2 for i in m.t)
            smooth_penalty = sum(
                0.05 * (m.action[i] - m.action[i - 1]) ** 2 for i in range(1, H)
            )
            # Scale weight per time step => Horizon length does not influence the optimum
            reg_penalty = sum((0.01 / H) * m.action[i] ** 2 for i in m.t)

            return temp_penalty + reg_penalty + smooth_penalty

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Solve the optimization model
        solver = pyo.SolverFactory(
            "ipopt", executable="/home/iai/ii6824/.local/bin/ipopt"
        )
        solver.solve(model, tee=False)

        # Extract the first optimal action
        action_first = pyo.value(model.action[0])

        # Clipping to [-1, +1] if numeric limits are exceeded
        action_clipped = np.clip(action_first, -1.0, 1.0)

        return np.array([action_clipped]), None
