import numpy as np
import torch
import deepxde as dde
from src.configs.case1.params import *
from src.configs.case1.pinn import rec, disks_c, disks_w, time_domain, spatial_time_domain

# Sample Points
omega_points = spatial_time_domain.random_points(8192)
rec_points = dde.geometry.GeometryXTime(rec, time_domain).random_boundary_points(512 // 3)
disk_c_points = None
for disk_c in disks_c:
    new_points = dde.geometry.GeometryXTime(disk_c, time_domain) \
        .random_boundary_points(512 // 3 // len(disks_c))
    if disk_c_points is None:
        disk_c_points = new_points
    else:
        disk_c_points = np.concatenate((
            disk_c_points, new_points
        ))
disk_w_points = None
for disk_w in disks_w:
    new_points = dde.geometry.GeometryXTime(disk_w, time_domain) \
        .random_boundary_points(512 // 3 // len(disks_w))
    if disk_w_points is None:
        disk_w_points = new_points
    else:
        disk_w_points = np.concatenate((
            disk_w_points, new_points
        ))
tot_points = np.concatenate((omega_points, rec_points, disk_c_points, disk_w_points))

# Area and numbers
area_omega = L * H - np.pi * (disk_rs_c[0] ** 2) * len(disk_centers_c) \
    - np.pi * (disk_rs_w[0] ** 2) * len(disk_centers_w)
num_omega = omega_points.shape[0]
len_rec = 2 * (L + H)
num_rec = rec_points.shape[0]
len_disk_c = 2 * np.pi * disk_rs_c[0] * len(disk_centers_c)
num_disk_c = disk_c_points.shape[0]
len_disk_w = 2 * np.pi * disk_rs_w[0] * len(disk_centers_w)
num_disk_w = disk_w_points.shape[0]

def loss_pfnn(x, T):
    T_x = dde.grad.jacobian(T, x, i=0, j=0)
    T_y = dde.grad.jacobian(T, x, i=0, j=1)
    T_t = dde.grad.jacobian(T, x, i=0, j=2)
    # variational form
    # a(u,u)
    start = 0
    a = area_omega * torch.mean(k * (T_x[start:start+num_omega, :] ** 2 
        + T_y[start:start+num_omega, :] ** 2))
    start += num_omega
    a += len_rec * torch.mean(T[start:start+num_rec, :] ** 2)
    start += num_rec
    a += len_disk_c * torch.mean(T[start:start+num_disk_c, :] ** 2)
    start += num_disk_c
    a += len_disk_w * torch.mean(T[start:start+num_disk_w, :] ** 2)
    start += num_disk_w
    # L(u)
    start = 0
    L = area_omega * torch.mean(-T_t[start:start+num_omega, :] * T[start:start+num_omega, :])
    start += num_omega
    L += len_rec * torch.mean(T[start:start+num_rec, :] * 1e-1)
    start += num_rec
    L += len_disk_c * torch.mean(T[start:start+num_disk_c, :] * 5.)
    start += num_disk_c
    L += len_disk_w * torch.mean(T[start:start+num_disk_w, :] * 1.)
    start += num_disk_w

    return (0.5 * a - L).reshape(1, 1)

initial_condition = lambda _: 1e-1