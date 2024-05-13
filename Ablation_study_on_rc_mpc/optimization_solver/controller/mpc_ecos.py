import cvxpy as cp
import numpy as np

class MPC:
    def __init__(self, control_horizon=12, weight=2.2):

        self.h = control_horizon

        self.Rie = 9.754103e-05
        self.Rea = 3.404517e-03
        self.Rwin = 2.122519e-03
        self.Ci = 1.097712e+08
        self.Ce = 4.736527e+11
        self.Ai = 7.076894e+01
        self.Ae = 1.865195e+02
        self.a = 3.095382e+01
        self.dt = 3600


        self.weight = weight

        # Define decision variables
        self.u = cp.Variable(self.h, nonneg=True)  # u is non-negative
        self.ti = cp.Variable(self.h)  # No sign restriction
        self.te = cp.Variable(self.h)  # No sign restriction
        self.delta = cp.Variable(self.h, nonneg=True)  # Deviation is non-negative

        self.ti_init=cp.Parameter(1)
        self.te_init = cp.Parameter(1)
        self.ta = cp.Parameter(self.h)  # No sign restriction
        self.sol_rad = cp.Parameter(self.h, nonneg=True)  # Solar radiation is non-negative
        self.int_gains = cp.Parameter(self.h, nonneg=True)  # Internal heat gains are non-negative
        self.LowerSetp =cp.Parameter(self.h)  # No sign restriction
        self.UpperSetp =  cp.Parameter(self.h)  # No sign restriction

        # Build constraints
        self.constraints = [self.u <= 1]  # In CVXPY, a scalar 1 is automatically expanded to match the dimensions of self.u.

        self.constraints.append(
            self.ti[0] == self.ti_init[0] + self.dt / self.Ci * ((self.ta[0] - self.ti_init[0]) / self.Rwin
                                                                + (self.te_init[0] - self.ti_init[0]) / self.Rie
                                                                + (7221 * self.u[0] + 2200)
                                                                + self.Ai * self.sol_rad[0]
                                                                + self.a * self.int_gains[0]))
        self.constraints.append(self.te[0] == self.te_init[0] + self.dt / self.Ce * ((self.ti_init[0] - self.te_init[0]) / self.Rie
                                                                                    + (self.ta[0] - self.te_init[0]) / self.Rea
                                                                                    + self.Ae * self.sol_rad[0]
                                                                                    ))

        for i in range(1,self.h):
            self.constraints.append(self.ti[i]==self.ti[i-1]+self.dt/self.Ci*( (self.ta[i]-self.ti[i-1])/self.Rwin
                                                                               +(self.te[i-1]-self.ti[i-1])/self.Rie
                                                                               +(7221 * self.u[i] + 2200)
                                                                               +self.Ai*self.sol_rad[i]
                                                                               +self.a*self.int_gains[i]))
            self.constraints.append(self.te[i] == self.te[i-1] + self.dt / self.Ce * ((self.ti[i-1] - self.te[i-1]) / self.Rie
                                                                                  + (self.ta[i] - self.te[i-1]) / self.Rea
                                                                                  + self.Ae * self.sol_rad[i]
                                                                                  ))
        for i in range(self.h):
            self.constraints.append(self.ti[i] >= self.LowerSetp[i] - self.delta[i])
            self.constraints.append(self.ti[i] <= self.UpperSetp[i] + self.delta[i])

    def forecast(self, current_states, disturbance, price, UpperSetp, LowerSetp):
        # Set the current values for variables
        init_ti, init_ta, init_I_s, init_Qint, init_Q_h = current_states
        init_te = (self.Rie * init_ta + self.Rea * init_ti) / (self.Rie + self.Rea)

        self.ti_init.value = [init_ti]
        self.te_init.value = [init_te]
        self.ta.value = disturbance[0]
        self.int_gains.value = disturbance[1]
        self.sol_rad.value = disturbance[2]
        self.LowerSetp.value = LowerSetp
        self.UpperSetp.value = UpperSetp

        # Define the objective function
        obj = cp.Minimize(self.weight * cp.sum(self.delta) + cp.sum(cp.multiply(price, self.u * 1.290 + 1.115)))

        # Create and solve the optimization problem
        prob = cp.Problem(obj, self.constraints)
        solver_args = {'verbose': True}  # 设置verbose参数以获取求解过程输出

        prob.solve(solver=cp.ECOS, max_iters=10000,solver_args=solver_args)
        convergence_iterations = prob.solver_stats.num_iters

        # Return the optimal value of the first control input
        return self.u.value[0],convergence_iterations
# Example usage
# mpc_cvxpy = MPC()
# current_states = [20,0,0,0,0]
# disturbance = [np.array([20]*12), np.array([5]*12), np.array([0]*12)]
# price = np.array([0.3]*12)
# UpperSetp = np.array([24]*12)
# LowerSetp = np.array([20]*12)
# u_opt = mpc_cvxpy.forecast(current_states, disturbance, price, UpperSetp, LowerSetp)
# print(u_opt)


