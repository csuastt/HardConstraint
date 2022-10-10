import torch
from torch import nn
import numpy as np
import deepxde as dde
from src.HC.hard_constraint import HardConstraintRobin2D
from src.HC.hard_constraint_collector import HardConstraintCollector
from src.HC.l_functions import LFunctionDisk, LFunctionRectangle
from src.HC.normal_function import NormalFunctionDisk, NormalFunctionRectangle
from src.configs.case1.pinn import rec, disks_c, disks_w
from src.utils.nn_wrapper import NNWrapper
from src.configs.case1.params import *

def pde_hc(x, u):
    p_1, p_2 = u[:, 1:2], u[:, 2:]
    T_x = dde.grad.jacobian(u, x, i=0, j=0)
    T_y = dde.grad.jacobian(u, x, i=0, j=1)
    T_t = dde.grad.jacobian(u, x, i=0, j=2)
    p_1_x = dde.grad.jacobian(u, x, i=1, j=0)
    p_2_y = dde.grad.jacobian(u, x, i=2, j=1)

    res = T_t - k * (p_1_x + p_2_y)
    p_1_res = p_1 - T_x
    p_2_res = p_2 - T_y
    return [res, p_1_res, p_2_res]


# Helper functions (input: x, a B x d tensor)
disks = disks_c + disks_w
X = rec.random_boundary_points(512)
for disk in disks:
    X = np.concatenate((X, disk.random_boundary_points(256)), axis=0)
# M function
beta = 4.0
M = lambda x: torch.logsumexp(-beta * x, dim=1, keepdim=True) / (-beta)
# Lambda functions
l_gamma_rec = LFunctionRectangle(X, rec, m_function=M)
l_gamma_disks_c = [LFunctionDisk(X, disk) for disk in disks_c]
l_gamma_disks_w = [LFunctionDisk(X, disk) for disk in disks_w]
# Normal functions
n_gamma_outer = NormalFunctionRectangle(H, L)
n_gamma_disks_c = [NormalFunctionDisk(center=disk_centers_c[i]) for i in range(len(disk_centers_c))]
n_gamma_disks_w = [NormalFunctionDisk(center=disk_centers_w[i]) for i in range(len(disk_centers_w))]

# hard constraints
hc_gamma_outer = HardConstraintRobin2D(
    n_gamma_outer, 
    NNWrapper(dde.nn.FNN([3] + 3 * [20] + [3], "tanh", "Glorot normal")),
    lambda _: 1.,
    lambda _: 1.,
    lambda _: 1e-1
)
hc_gamma_disks_c = [
    HardConstraintRobin2D(
        n_gamma_disk, 
        NNWrapper(dde.nn.FNN([3] + 3 * [20] + [3], "tanh", "Glorot normal")),
        lambda _: 1.,
        lambda _: 1.,
        lambda _: 5.
    ) for n_gamma_disk in n_gamma_disks_c
]
hc_gamma_disks_w = [
    HardConstraintRobin2D(
        n_gamma_disk, 
        NNWrapper(dde.nn.FNN([3] + 3 * [20] + [3], "tanh", "Glorot normal")),
        lambda _: 1.,
        lambda _: 1.,
        lambda _: 1.
    ) for n_gamma_disk in n_gamma_disks_w
]

# model
class HCNN(nn.Module):
    """
    Hard constraint model.
    """
    def __init__(self, path_prefix="") -> None:
        super(HCNN, self).__init__()
        # NNs
        self.hc_gamma_outer = hc_gamma_outer
        self.hc_gamma_disks = nn.ModuleList(hc_gamma_disks_c + hc_gamma_disks_w)
        self.N_main = NNWrapper(dde.nn.FNN([3] + 4 * [50] + [3], "tanh", "Glorot normal"))
        if path_prefix != "":
            self.load(path_prefix)
        # hard constraint for each components
        self.HCC_T = HardConstraintCollector(
            0, M, [l_gamma_rec] + l_gamma_disks_c + l_gamma_disks_w,
            [
                hc_gamma_outer.get_u
            ] + [
                hc_gamma_disk.get_u for hc_gamma_disk in self.hc_gamma_disks
            ], self.N_main
        )
        self.HCC_p_1 = HardConstraintCollector(
            1, M, [l_gamma_rec] + l_gamma_disks_c + l_gamma_disks_w,
            [
                hc_gamma_outer.get_p_1
            ] + [
                hc_gamma_disk.get_p_1 for hc_gamma_disk in self.hc_gamma_disks
            ], self.N_main
        )
        self.HCC_p_2 = HardConstraintCollector(
            2, M, [l_gamma_rec] + l_gamma_disks_c + l_gamma_disks_w,
            [
                hc_gamma_outer.get_p_2
            ] + [
                hc_gamma_disk.get_p_2 for hc_gamma_disk in self.hc_gamma_disks
            ], self.N_main
        )
    
    def save(self, path_prefix: str):
        torch.save(self.N_main, path_prefix + "hc_main.pth")
        self.hc_gamma_outer.save(path_prefix, "hc_g_o.pth")
        for i in range(len(self.hc_gamma_disks)):
            self.hc_gamma_disks[i].save(path_prefix, "hc_g_d%d.pth"%i)

    def load(self, path_prefix: str):
        self.N_main = torch.load(path_prefix + "hc_main.pth")
        self.hc_gamma_outer.load(path_prefix, "hc_g_o.pth")
        for i in range(len(self.hc_gamma_disks)):
            self.hc_gamma_disks[i].load(path_prefix, "hc_g_d%d.pth"%i)

    def forward(self, x) -> torch.Tensor:
        self.N_main.clear_res()
        self.hc_gamma_outer.clear_res()
        for hc_gamma_disk in self.hc_gamma_disks:
            hc_gamma_disk.clear_res()
        time_factor = torch.exp(-10 * x[:, 2:3])
        return torch.cat((
            self.HCC_T(x) * (1 - time_factor) + 1e-1 * time_factor,
            self.HCC_p_1(x),
            self.HCC_p_2(x)
        ), dim=1)
