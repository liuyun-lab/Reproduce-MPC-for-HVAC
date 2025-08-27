import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layers(x)

def ann_mpc(
    current_temp, disturbances, setpoints, price, weight, horizon=12, epsilon=0.5,
    model=None,
    x_scaler=joblib.load("ANN/x_scaler.save"),
    y_scaler=joblib.load("ANN/y_scaler.save"),
    lr=0.95, momentum =0.9, n_iter=100, tolerance=1e-4, patience=5
):
    """
    ANN-based MPC using PSGD for optimization with early stopping.

    Optimizes a sequence of future control actions (u) over the prediction horizon,
    using a trained ANN to predict indoor temperature. The cost includes energy cost and
    thermal discomfort.

    Args:
        current_temp (float): Current indoor temperature.
        disturbances (dict): Future disturbance sequences, each of length 'horizon'. Keys: 'Ta', 'Isol', 'Qint'.
        setpoints (dict): Lower and upper temperature setpoints for the horizon. Keys: 'Lower', 'Upper'.
        price (array-like): Future electricity prices for the horizon.
        weight (float): Penalty weight for thermal discomfort.
        horizon (int): Prediction/optimization horizon.
        epsilon (float): Bandwidth for soft temperature constraint.
        model (nn.Module, optional): Trained ANN model. If None, loads default model.
        x_scaler, y_scaler: Scalers for input and output normalization.
        lr (float): Learning rate for SGD.
        n_iter (int): Number of optimization iterations.
        tolerance (float): Early stopping threshold.
        patience (int): Early stopping patience.

    Returns:
        float: The first optimized control action for the current step.
    """
    if model is None:
        model = MLP(input_size=5)
        model.load_state_dict(torch.load("ANN/best_mlp.pth"))
        model.eval()

    device = torch.device('cpu')
    u = torch.zeros(horizon, requires_grad=True, device=device)  # Optimization variable

    optimizer = torch.optim.SGD([u], lr=lr,momentum = momentum)
    #optimizer = torch.optim.Adam([u], lr=lr)

    prev_loss = float('inf')
    no_improvement_counter = 0
    min_loss = float('inf')
    best_u = None

    # Prepare scaler parameters as tensors
    x_min = torch.tensor(x_scaler.data_min_, dtype=torch.float32, device=device)
    x_max = torch.tensor(x_scaler.data_max_, dtype=torch.float32, device=device)
    y_min = torch.tensor(y_scaler.data_min_, dtype=torch.float32, device=device)
    y_max = torch.tensor(y_scaler.data_max_, dtype=torch.float32, device=device)

    for it in range(n_iter):
        optimizer.zero_grad()
        Ti_pred = [torch.tensor([current_temp], dtype=torch.float32, device=device)]
        delta_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
        cost_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
        for k in range(horizon):
            Qh_k = u[k] * 7221 + 2220
            # Build input as tensor
            ann_input = torch.cat([
                Ti_pred[-1],
                torch.tensor([disturbances['Ta'][k]], dtype=torch.float32, device=device),
                torch.tensor([disturbances['Isol'][k]], dtype=torch.float32, device=device),
                Qh_k.unsqueeze(0),
                torch.tensor([disturbances['Qint'][k]], dtype=torch.float32, device=device)
            ]).unsqueeze(0)  # shape (1, 5)
            # Normalize
            ann_input_norm = (ann_input - x_min) / (x_max - x_min)
            Ti_next_norm = model(ann_input_norm)
            # Denormalize
            Ti_next = Ti_next_norm * (y_max - y_min) + y_min
            Ti_pred.append(Ti_next.squeeze(0))
            lower_bound = torch.tensor(setpoints['Lower'][k] + epsilon, dtype=torch.float32, device=device)
            upper_bound = torch.tensor(setpoints['Upper'][k] - epsilon, dtype=torch.float32, device=device)
            delta = F.relu(lower_bound - Ti_next) + F.relu(Ti_next - upper_bound)
            delta_sum = delta_sum + delta
            cost_sum = cost_sum + price[k] * (u[k] * 1.290 + 1.115)
        obj = weight * delta_sum + cost_sum
        #print(f"iter {it}: u = {u.detach().cpu().numpy()}, cost_sum={cost_sum.item():.3f}, weight*delta_sum={(weight * delta_sum).item():.3f}, obj={obj.item():.3f}")
        current_loss = obj.item()
        obj.backward()
        #print(f"Gradient of u: {u.grad}")
        optimizer.step()
        with torch.no_grad():
            u.clamp_(0, 1)

        # Early stopping logic
        if current_loss >= prev_loss - tolerance:
            no_improvement_counter += 1
            if no_improvement_counter >= patience:
                break
        else:
            no_improvement_counter = 0
        prev_loss = current_loss
        if current_loss < min_loss:
            min_loss = current_loss
            best_u = u.clone()

    # Return the first control action from the best found u
    if best_u is not None:
        #print(f"iter {it}: u = {best_u.detach().cpu().numpy()[0]}")
        return best_u.detach().cpu().numpy()[0]
    else:
        #print(f"iter {it}: u = {u.detach().cpu().numpy()[0]}")
        return u.detach().cpu().numpy()[0]