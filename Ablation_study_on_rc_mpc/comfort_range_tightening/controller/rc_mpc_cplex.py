from docplex.mp.model import Model
import numpy as np


class MPC:
    def __init__(self,control_horizon=12,weight=0.3):

        # self.Rie=2.680219e-04
        # self.Rea=1.571491e-02
        # self.Rwin=3.406053e-03
        # self.Ci=6.129641e+07
        # self.Ce=1.600232e+10
        # self.Ai=3.240800e+01
        # self.Ae=3.176265e+01

        self.Rie = 9.754103e-05
        self.Rea = 3.404517e-03
        self.Rwin = 2.122519e-03
        self.Ci = 1.097712e+08
        self.Ce = 4.736527e+11
        self.Ai = 7.076894e+01
        self.Ae = 1.865195e+02
        self.a=3.095382e+01
        self.dt=3600


        self.h=control_horizon
        self.weight=weight

        self.m=Model(name='mpc')
        self.u = self.m.continuous_var_list(keys=self.h, lb=0, ub=1, name='heat pump modulating signal')

        self.ti=self.m.continuous_var_list(keys=self.h+1,name='indoor temperature')
        self.te = self.m.continuous_var_list(keys=self.h+1, name='wall temperature')
        self.ta=self.m.continuous_var_list(keys=self.h,name='outdoor temperature')
        self.sol_rad=self.m.continuous_var_list(keys=self.h,lb=0,name='solar radiation')
        self.int_gains=self.m.continuous_var_list(keys=self.h,lb=0,name='internal heat gain')
        self.delta=self.m.continuous_var_list(keys=self.h,lb=0,name='indoor temperature deviation')
        self.LowerSetp=self.m.continuous_var_list(keys=self.h,name='Lower temperature setpoints')
        self.UpperSetp=self.m.continuous_var_list(keys=self.h,name='Upper temperature setpoints')




        for i in range(self.h):
            self.m.add_constraint(self.ti[i+1]==self.ti[i]+self.dt/self.Ci*(   (self.ta[i]-self.ti[i])/self.Rwin
                                                                               +(self.te[i]-self.ti[i])/self.Rie
                                                                               +(7221 * self.u[i] + 2200)
                                                                               +self.Ai*self.sol_rad[i]
                                                                               +self.a*self.int_gains[i]) )

            self.m.add_constraint(self.te[i+1]==self.te[i]+self.dt/self.Ce*(   (self.ti[i]-self.te[i])/self.Rie
                                                                               +(self.ta[i]-self.te[i])/self.Rea
                                                                               +self.Ae*self.sol_rad[i]
                                                                               ))

            self.m.add_constraint(self.ti[i+1]>=self.LowerSetp[i]-self.delta[i])
            self.m.add_constraint(self.ti[i+1]<=self.UpperSetp[i]+self.delta[i])






    def forecast(self,current_states,disturbance,price,UpperSetp,LowerSetp):

        ta=disturbance[0]
        int_gains=disturbance[1]
        sol_rad=disturbance[2]
        init_ti,init_ta,init_I_s,init_Qint,init_Q_h=current_states
        init_te=(self.Rie*init_ta+self.Rea*init_ti)/(self.Rie+self.Rea)



        self.ti[0].ub = init_ti
        self.ti[0].lb = init_ti
        self.te[0].ub= init_te
        self.te[0].lb=init_te

        self.m.change_var_upper_bounds(self.ta,ta)
        self.m.change_var_lower_bounds(self.ta,ta)

        self.m.change_var_upper_bounds(self.sol_rad,sol_rad)
        self.m.change_var_lower_bounds(self.sol_rad,sol_rad)

        self.m.change_var_upper_bounds(self.int_gains,int_gains)
        self.m.change_var_lower_bounds(self.int_gains,int_gains)

        self.m.change_var_upper_bounds(self.UpperSetp,UpperSetp)
        self.m.change_var_lower_bounds(self.UpperSetp,UpperSetp)

        self.m.change_var_upper_bounds(self.LowerSetp, LowerSetp)
        self.m.change_var_lower_bounds(self.LowerSetp, LowerSetp)

#         self.m.change_var_upper_bounds(self.u,[0.3, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# )
#         self.m.change_var_lower_bounds(self.u, [0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#                                        )

        obj=0
        obj+=self.weight*sum(self.delta[t] for t in range(self.h))
        obj+=sum(price[t]*(self.u[t] * 1.290 + 1.115) for t in range(self.h))

        self.m.minimize(obj)

        solution=self.m.solve(log_output=False)

        u_opt=self.u[0].solution_value

        return u_opt





