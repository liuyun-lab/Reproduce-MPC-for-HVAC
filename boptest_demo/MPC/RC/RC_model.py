import numpy as np

def simulate_3r2c_multistep(u_seq, theta, Ti0):
    """
    Multistep (free-run) simulation for the 3R2C model.
    At each step, use the model's previous predicted state as the initial state.
    u_seq: N×4 input array, columns are [Ta, Isol, Qh, Qint]
    theta: parameter vector [Rwin, Rea, Rie, Ci, Ce, Ai, Ae, K]
    Ti0: initial indoor temperature
    Ta0: initial outdoor temperature
    Returns: Ti_sim (N-length array of simulated indoor temperature)
    """
    Rwin, Rea, Rie, Ci, Ce, Ai, Ae, K = theta
    N = u_seq.shape[0]
    dt = 3600  # 1 hour in seconds

    # Initial wall temperature
    Ta0 = u_seq[0, 0]
    Te0 = (Rie * Ti0 + Rea * Ta0) / (Rie + Rea)

    # State vector: x[:, 0] = Te, x[:, 1] = Ti
    x = np.zeros((N+1, 2))
    x[0, 0] = Te0  # Initial wall temperature
    x[0, 1] = Ti0  # Initial indoor temperature

    # Discrete-time state-space matrices
    A = np.array([
        [-(1/Ce)*(1/Rea + 1/Rie), 1/(Ce*Rie)],
        [1/(Ci*Rie), -(1/Ci)*(1/Rwin + 1/Rie)]
    ])
    B = np.array([
        [1/(Ce*Rea), Ae/Ce, 0, 0],
        [1/(Ci*Rwin), Ai/Ci, 1/Ci, K/Ci]
    ])
    A_d = np.eye(2) + dt * A
    B_d = dt * B

    # Multi-step simulation loop
    for k in range(N):
        x[k+1] = A_d @ x[k] + B_d @ u_seq[k]
    return x[1:, 1]  # Return the simulated indoor temperature sequence (Ti)


def simulate_3r2c_onestep(u_seq, theta, Ti_measured):
    """
    One-step-ahead simulation for 3R2C model.
    At each step, use the measured indoor temperature as the initial state.
    u_seq: N×4 input array, columns are [Ta, Isol, Qh, Qint]
    theta: parameter vector [Rwin, Rea, Rie, Ci, Ce, Ai, Ae, K]
    Ti_measured: N-length array of measured indoor temperature
    Returns: Ti_sim (N-length array of simulated indoor temperature)
    """
    Rwin, Rea, Rie, Ci, Ce, Ai, Ae, K = theta
    N = u_seq.shape[0]
    dt = 3600  # 1 hour in seconds

    # State-space matrices
    A = np.array([
        [-(1/Ce)*(1/Rea + 1/Rie), 1/(Ce*Rie)],
        [1/(Ci*Rie), -(1/Ci)*(1/Rwin + 1/Rie)]
    ])
    B = np.array([
        [1/(Ce*Rea), Ae/Ce, 0, 0],
        [1/(Ci*Rwin), Ai/Ci, 1/Ci, K/Ci]
    ])
    A_d = np.eye(2) + dt * A
    B_d = dt * B

    Ta_seq = u_seq[:, 0]

    Ti_sim = np.zeros(N)
    Te_sim = np.zeros(N)
    Ti_sim[0] = Ti_measured[0]
    Te_sim[0] = (Rie * Ti_measured[0] + Rea * Ta_seq[0]) / (Rie + Rea)

    for k in range(1, N):
        # At each step, use the measured indoor temperature and calculated wall temperature
        x_prev = np.array([Te_sim[k - 1], Ti_measured[k - 1]])

        u_prev = u_seq[k-1]
        x_next = A_d @ x_prev + B_d @ u_prev
        Te_sim[k] = x_next[0]
        Ti_sim[k] = x_next[1]
    return Ti_sim
