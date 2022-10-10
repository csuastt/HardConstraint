import math
import numpy as np
import torch
import deepxde as dde
from src.configs.case3.params import *
from src.configs.case3.pinn import spatial_time_domain, reference_solution

# Sample Points
omega_points = spatial_time_domain.random_points(1000)
boundary_points = spatial_time_domain.random_boundary_points(100)
tot_points = np.concatenate((omega_points, boundary_points))

# Area and numbers
volume_omega = math.pi ** (d/2) / math.gamma(d/2 + 1)
num_omega = omega_points.shape[0]
area_boundary = 2 * math.pi ** (d/2) / math.gamma(d/2)
num_boundary = boundary_points.shape[0]

def loss_pfnn(x, T):
    T_t = dde.grad.jacobian(T, x, i=0, j=d)
    nabla_dot_nabla_T = 0.
    for i in range(d):
        nabla_dot_nabla_T += dde.grad.jacobian(T, x, i=0, j=i) ** 2
    squared_norm = torch.sum(x[:, :d] ** 2, dim=1, keepdim=True)
    f = -alpha * squared_norm * torch.exp(0.5 * squared_norm + x[:, d:d+1])

    # variational form
    # a(u,u)
    start = 0
    a = volume_omega * torch.mean(alpha * nabla_dot_nabla_T)
    start += num_omega
    # L(u)
    start = 0
    L = volume_omega * torch.mean((f[start:start+num_omega, :]-T_t[start:start+num_omega, :]) 
        * T[start:start+num_omega, :])
    start += num_omega
    L += area_boundary * torch.mean(alpha * T[start:start+num_boundary, :] 
        * reference_solution_pt(x[start:start+num_boundary, :]))
    start += num_boundary

    return (0.5 * a - L).reshape(1, 1)


def reference_solution_pt(x):
    squared_norm = torch.sum(x[:, :d] ** 2, dim=1, keepdim=True)
    return torch.exp(0.5 * squared_norm + x[:, d:d+1])