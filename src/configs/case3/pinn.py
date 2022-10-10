import numpy as np
import torch
import deepxde as dde
from src.configs.case3.params import *

spatial_domain = dde.geometry.Hypersphere([0] * d, 1)
time_domain = dde.geometry.TimeDomain(0, 1)
spatial_time_domain = dde.geometry.GeometryXTime(spatial_domain, time_domain)


def pde_pinn(x, T):
    T_t = dde.grad.jacobian(T, x, i=0, j=d)
    delta_T = 0.
    for i in range(d):
        delta_T += dde.grad.hessian(T, x, i=i, j=i)
    squared_norm = torch.sum(x[:, :d] ** 2, dim=1, keepdim=True)
    f = -alpha * squared_norm * torch.exp(0.5 * squared_norm + x[:, d:d+1])
    return T_t - alpha * delta_T - f


def reference_solution(x):
    squared_norm = np.sum(x[:, :d] ** 2, axis=1, keepdims=True)
    return np.exp(0.5 * squared_norm + x[:, d:d+1])


bc = dde.icbc.NeumannBC(
    spatial_time_domain, reference_solution,
    lambda _, on_bc: on_bc
)


ic = dde.icbc.IC(
    spatial_time_domain, reference_solution, 
    lambda _, on_initial: on_initial
)

ic_bcs = [bc, ic]

# parameters for PINN
# moving average in learning rate annealing
lr_alpha = 0.1
# num pdes
num_pdes = 1
num_bcs = len(ic_bcs)