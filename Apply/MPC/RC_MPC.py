from docplex.mp.model import Model
import numpy as np
import json
def rc_mpc(
    current_temp, disturbances, setpoints, price, weight, theta=None, horizon=12, epsilon=0.5
):
    """
    RC-based MPC using CPLEX (docplex).

    Args:
        current_temp (float): Current indoor temperature.
        disturbances (dict): Future disturbance sequences, each of length 'horizon'. Keys: 'Ta', 'Isol', 'Qint'.
        setpoints (dict): Lower and upper temperature setpoints for the horizon. Keys: 'Lower', 'Upper'.
        price (array-like): Future electricity prices for the horizon.
        weight (float): Penalty weight for thermal discomfort.
        theta (list): RC model parameters [Rwin, Rea, Rie, Ci, Ce, Ai, Ae, K].
        horizon (int): Prediction/optimization horizon.
        epsilon (float): Bandwidth for soft temperature constraint.

    Returns:
        float: The first optimized control action for the current step.
    """
    if theta is None:
        with open("RC/identified_params.json", "r") as f:
            params_dict = json.load(f)
        param_names = ["Rwin", "Rea", "Rie", "Ci", "Ce", "Ai", "Ae", "K"]
        theta = [params_dict[name] for name in param_names]


    Rwin, Rea, Rie, Ci, Ce, Ai, Ae, K = theta
    dt = 3600

    m = Model(name='rc_mpc')
    u = m.continuous_var_list(keys=horizon, lb=0, ub=1, name='u')
    delta = m.continuous_var_list(keys=horizon, lb=0, name='delta')
    ti = m.continuous_var_list(keys=horizon+1, name='ti')
    te = m.continuous_var_list(keys=horizon+1, name='te')

    # Initial wall temperature
    Ta0 = disturbances['Ta'][0]
    Te0 = (Rie * current_temp + Rea * Ta0) / (Rie + Rea)
    m.add_constraint(ti[0] == current_temp)
    m.add_constraint(te[0] == Te0)

    # RC model prediction and constraints
    for k in range(horizon):
        # RC model state update
        m.add_constraint(
            ti[k+1] == ti[k] + dt/Ci * (
                (disturbances['Ta'][k] - ti[k]) / Rwin +
                (te[k] - ti[k]) / Rie +
                (u[k] * 7220 + 2220) +
                Ai * disturbances['Isol'][k] +
                K * disturbances['Qint'][k]
            )
        )
        m.add_constraint(
            te[k+1] == te[k] + dt/Ce * (
                (ti[k] - te[k]) / Rie +
                (disturbances['Ta'][k] - te[k]) / Rea +
                Ae * disturbances['Isol'][k]
            )
        )
        # Soft temperature constraints
        lower_bound = setpoints['Lower'][k] + epsilon
        upper_bound = setpoints['Upper'][k] - epsilon
        m.add_constraint(ti[k+1] >= lower_bound - delta[k])
        m.add_constraint(ti[k+1] <= upper_bound + delta[k])

    # Objective: energy cost + discomfort penalty
    obj = m.sum(price[k] * (u[k] * 1.290 + 1.115) + weight * delta[k] for k in range(horizon))
    m.minimize(obj)

    solution = m.solve(log_output=False)
    u_opt = u[0].solution_value
    return u_opt

