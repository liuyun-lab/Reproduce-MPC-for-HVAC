import casadi as ca
import numpy as np

class MPC:
    def __init__(self, control_horizon=12, weight=2.2):
        self.Rie = 9.754103e-05
        self.Rea = 3.404517e-03
        self.Rwin = 2.122519e-03
        self.Ci = 1.097712e+08
        self.Ce = 4.736527e+11
        self.Ai = 7.076894e+01
        self.Ae = 1.865195e+02
        self.a = 3.095382e+01
        self.dt = 3600
        self.h = control_horizon
        self.weight = weight

        # Define decision variables
        self.u = ca.MX.sym('u', self.h)
        self.ti = ca.MX.sym('ti', self.h)
        self.te = ca.MX.sym('te', self.h)
        self.ti_init = ca.MX.sym('ti_init')
        self.te_init = ca.MX.sym('te_init')
        self.ta = ca.MX.sym('ta', self.h)
        self.sol_rad = ca.MX.sym('sol_rad', self.h)
        self.int_gains = ca.MX.sym('int_gains', self.h)
        self.delta = ca.MX.sym('delta', self.h)
        self.LowerSetp = ca.MX.sym('LowerSetp', self.h)
        self.UpperSetp = ca.MX.sym('UpperSetp', self.h)

        self.lbg=[]
        self.ubg=[]

        # Build constraints
        self.constraints = []
        self.constraints.append(
            self.ti[0]-self.ti_init -self.dt / self.Ci * ((self.ta[0] - self.ti_init[0]) / self.Rwin
                                                                + (self.te_init[0] - self.ti_init[0]) / self.Rie
                                                                + (7221 * self.u[0] + 2200)
                                                                + self.Ai * self.sol_rad[0]
                                                                + self.a * self.int_gains[0]))
        self.lbg.append(0)
        self.ubg.append(0)

        self.constraints.append(
            self.te[0] -self.te_init - self.dt / self.Ce * ((self.ti_init[0] - self.te_init[0]) / self.Rie
                                                                 + (self.ta[0] - self.te_init[0]) / self.Rea
                                                                 + self.Ae * self.sol_rad[0]
                                                                 ))
        self.lbg.append(0)
        self.ubg.append(0)

        for i in range(1, self.h):
            self.constraints.append(
                self.ti[i] - self.ti[i - 1] - self.dt / self.Ci * ((self.ta[i] - self.ti[i - 1]) / self.Rwin
                                                                    + (self.te[i - 1] - self.ti[i - 1]) / self.Rie
                                                                    + (7221 * self.u[i] + 2200)
                                                                    + self.Ai * self.sol_rad[i]
                                                                    + self.a * self.int_gains[i]))
            self.lbg.append(0)
            self.ubg.append(0)

            self.constraints.append(
                self.te[i] - self.te[i - 1] - self.dt / self.Ce * ((self.ti[i - 1] - self.te[i - 1]) / self.Rie
                                                                    + (self.ta[i] - self.te[i - 1]) / self.Rea
                                                                    + self.Ae * self.sol_rad[i]
                                                                    ))
            self.lbg.append(0)
            self.ubg.append(0)

        for i in range(self.h):
            self.constraints.append(self.ti[i]-(self.LowerSetp[i] - self.delta[i]))
            self.lbg.append(0)
            self.ubg.append(ca.inf)
            self.constraints.append((self.UpperSetp[i] + self.delta[i])-self.ti[i])
            self.lbg.append(0)
            self.ubg.append(ca.inf)

    def forecast(self, current_states, disturbance, price, UpperSetp, LowerSetp):
        # Set the current values for variables
        init_ti, init_ta, init_I_s, init_Qint, init_Q_h = current_states
        init_te = (self.Rie * init_ta + self.Rea * init_ti) / (self.Rie + self.Rea)
        ta, int_gains, sol_rad = disturbance

        # Objective function
        obj = self.weight * ca.sum1(self.delta) + ca.dot(price, ca.mtimes(self.u, 1.290) + 1.115)

        # Build and solve the optimization problem
        opt_variables = ca.vertcat(self.u, self.ti, self.ti_init, self.ta, self.sol_rad, self.int_gains, self.delta, self.LowerSetp, self.UpperSetp,self.te_init,self.te)
        nlp = {'x': opt_variables, 'f': obj, 'g': ca.vertcat(*self.constraints)}
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp,opts)

        lbx = ca.DM.zeros(opt_variables.shape[0])
        ubx = ca.DM.zeros(opt_variables.shape[0])

        # Set variable bound
        lbx[0:self.h] = 0  # Lower bound and upper bound for u
        ubx[0:self.h] = 1

        lbx[self.h:self.h*2] = -ca.inf  # Lower bound and upper bound for int_temp
        ubx[self.h:self.h*2] = ca.inf

        lbx[self.h*2] = init_ti  # Lower bound and upper bound for init_temp
        ubx[self.h*2] = init_ti

        lbx[self.h*2+1:self.h*3+1] = ta  # Lower bound and upper bound for ta
        ubx[self.h*2+1:self.h*3+1] = ta
        lbx[self.h*3+1:self.h*4+1] = sol_rad  # Lower bound and upper bound for sol_rad
        ubx[self.h*3+1:self.h*4+1] = sol_rad
        lbx[self.h*4+1:self.h*5+1] = int_gains  # Lower bound and upper bound for init_gains
        ubx[self.h*4+1:self.h*5+1] = int_gains

        ubx[self.h * 5 + 1:self.h * 6 + 1] = 0 # Lower bound and upper bound for delta
        ubx[self.h * 5 + 1:self.h * 6 + 1] = ca.inf

        lbx[self.h*6+1:self.h*7+1] = LowerSetp  # Lower bound and upper bound for LowerSetp
        ubx[self.h*6+1:self.h*7+1] = LowerSetp
        lbx[self.h*7+1:self.h*8+1] = UpperSetp  # Lower bound and upper bound for UpperSetp
        ubx[self.h*7+1:self.h*8+1] = UpperSetp

        lbx[self.h * 8 + 1] = init_te  # Lower bound and upper bound for init_temp
        ubx[self.h * 8 + 1] = init_te
        lbx[self.h*8+2:self.h*9+2] =  -ca.inf   # Lower bound and upper bound for init_temp
        ubx[self.h*8+2:self.h*9+2] =  ca.inf


        # Solve
        sol = solver(lbx=lbx, ubx=ubx, lbg=self.lbg, ubg=self.ubg)
        u_opt = np.clip(float(sol['x'][0]), 0, 1)
        convergence_iterations = sol.stats()['iter_count']

        return u_opt,convergence_iterations

# Example usage
# mpc_cvxpy = MPC()
# current_states = [20,0,0,0,0]
# disturbance = [np.array([20]*12), np.array([5]*12), np.array([0]*12)]
# price = np.array([0.3]*12)
# UpperSetp = np.array([24]*12)
# LowerSetp = np.array([20]*12)
# u_opt = mpc_cvxpy.forecast(current_states, disturbance, price, UpperSetp, LowerSetp)
# print(u_opt)

