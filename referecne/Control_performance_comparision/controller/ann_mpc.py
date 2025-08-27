import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Union
import matplotlib.pyplot as plt
import random
import torch.optim as optim
from torch.autograd import Variable
import joblib
import json
import requests
import argparse
import os


class MLP(nn.Module):
    def __init__(self,input_size,input_normalization_minimum,input_normalization_maximum,output_low_limit,output_high_limit):
        super(MLP, self).__init__()


        self.input_normalization_minimum = torch.tensor(input_normalization_minimum, dtype=torch.float32)
        self.input_normalization_maximum = torch.tensor(input_normalization_maximum, dtype=torch.float32)
        self.output_high_limit=torch.tensor(output_high_limit, dtype=torch.float32)
        self.output_low_limit=torch.tensor(output_low_limit, dtype=torch.float32)

        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def load_model(self):
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'ann_model.pth')

        self.load_state_dict(torch.load(model_path)["model_state_dict"])
    def normalize_input(self, data):

        normalized_data = (data - self.input_normalization_minimum) / \
                          (self.input_normalization_maximum - self.input_normalization_minimum)

        return normalized_data
    def normalize_output(self, output):

        normalize_output = (output - self.output_low_limit) / \
                          (self.output_high_limit - self.output_low_limit)

        return normalize_output
    def inverse_normalize_data(self, normalized_data_tensor):
        original_data = normalized_data_tensor * (self.output_high_limit - self.output_low_limit) + self.output_low_limit
        return original_data

    def forward(self, x):
        x_normalized = self.normalize_input(x)
        return self.normalize_output(self.layers(x_normalized))
    def step(self, x):
        x_normalized = self.normalize_input(x)
        return self.layers(x_normalized)

    def step_with_u(self,x,u, aux):
        aux_part1, aux_part2 = aux[0:1], aux[1:]

        if u.dim() == 0:
            u = u.unsqueeze(0)
        if x.dim() == 0:
            x = x.unsqueeze(0)
        input=torch.cat((aux_part1, u, x,aux_part2), dim=0)
        return self.step(input)

class MPC:
    def __init__(self,control_horizon=12,weight=50):
        self.h=control_horizon
        self.hist_u = torch.full((self.h, 1),0, dtype=torch.float32,requires_grad=True)
        self.mlp=MLP(5,[-9.1000,  0.0000,  4.9050,  0.0000,  0.0000],
                        [  29.85, 14187.3652,   48.78613256, 219,  862],
                     4.904992,
                     48.78613256)
        self.mlp.load_model()
        self.tolerance = 0.0001
        self.time_weights = torch.linspace(1, 0.95, self.h).reshape(self.h, 1)
        self.weight=weight

    def forecast(self,current_temp,disturbance,price,temperature_setpoints,UpperSetp,LowerSetp):

        UpperSetp_tensor=torch.tensor(UpperSetp,dtype=torch.float32).reshape(-1,1)
        LowerSetp_tensor=torch.tensor(LowerSetp,dtype=torch.float32).reshape(-1,1)
        u_data = torch.cat((self.hist_u[1:].detach(), self.hist_u[:1].detach()), dim=0)
        # print(self.hist_u)
        u = u_data.requires_grad_(True)
        optimizer = optim.SGD([u], lr=0.95, momentum=0.9)
        epochs=100

        cl_lower_bounds = torch.zeros((self.h, 1), dtype=torch.float32)
        cl_upper_bounds = torch.full((self.h, 1), 1, dtype=torch.float32)

        temperature_setpoints_tensor=torch.tensor(temperature_setpoints,dtype=torch.float32).reshape(-1,1)
        price_tensor=torch.tensor(price,dtype=torch.float32).reshape(-1,1)

        min_loss = float('inf')
        best_u = 0

        for j in range(epochs):

            current_state = torch.tensor(current_temp)

            temp_preds = torch.zeros((self.h, 1), dtype=torch.float32)
            prev_loss = float('inf')
            no_improvement_counter = 0

            for t in range(self.h):
                rolling_input = torch.tensor(disturbance[t],dtype=torch.float32)
                # if u[t]==0.00:
                #     current_state = self.mlp.step_with_u(current_state, u[t] *7221, rolling_input)
                # else:
                current_state = self.mlp.step_with_u(current_state,u[t] * 7221 + 2220,rolling_input)
                # current_state = self.mlp.step_with_u(current_state, u[t] * 7221 + 2220, rolling_input)
                temp_preds[t, :] = current_state

            optimizer.zero_grad()

            gradient, cost = self.compute_gradient_with_penalty(u, temp_preds, temperature_setpoints_tensor,
                                                                cl_lower_bounds,
                                                                cl_upper_bounds,price_tensor,UpperSetp_tensor,LowerSetp_tensor)

            current_loss = cost.item()

            if current_loss >= prev_loss - self.tolerance:
                no_improvement_counter += 1
                if no_improvement_counter >= 5:
                    break
            else:
                no_improvement_counter = 0

            prev_loss = current_loss
            # if j%10==0:
            #     print(f'epoch:{j},loss:{current_loss}')

            u.grad = gradient
            optimizer.step()

            upper_limit = cl_upper_bounds
            lower_limit = cl_lower_bounds
            u.data = torch.where(u.data > upper_limit, upper_limit, u.data)
            u.data = torch.where(u.data < lower_limit, lower_limit, u.data)
            if current_loss < min_loss:
                min_loss = current_loss
                best_u = u.clone()  

        self.hist_u = u

        action=u[0].item()

        return action

    def compute_gradient_with_penalty(self, control_signal, temperatures, temperature_setpoints, min_control, max_control,
                                      price,UpperSetp,LowerSetp):

        price_cost = torch.sum((control_signal * 1.290 + 1.115) * price)



        deviation_lower =  torch.relu(LowerSetp - temperatures)

       
        deviation_upper = torch.relu(temperatures - UpperSetp)


        temperature_deviation = torch.sum((deviation_lower  + deviation_upper ))



        cost =(price_cost+self.weight*temperature_deviation)
        cost.backward()


        gradient = control_signal.grad.clone()
        epsilon=1e-9
        gradient += epsilon * (1 / (max_control - control_signal+epsilon) + 1 / (control_signal - min_control+epsilon))

        return gradient, cost
