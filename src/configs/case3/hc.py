import torch
from torch import nn
import deepxde as dde
from src.HC.hard_constraint import HardConstraintNeumannND
from src.HC.normal_function import NormalFunctionSphere
from src.utils.nn_wrapper import NNWrapper
from src.configs.case3.params import *

def pde_hc(x, u):
    ps = [u[:, i+1:i+2] for i in range(d)]
    T_xs = [dde.grad.jacobian(u, x, i=0, j=i) for i in range(d)]
    ps_xs = [dde.grad.jacobian(u, x, i=i+1, j=i) for i in range(d)]
    T_t = dde.grad.jacobian(u, x, i=0, j=d)
    squared_norm = torch.sum(x[:, :d] ** 2, dim=1, keepdim=True)
    f = -alpha * squared_norm * torch.exp(0.5 * squared_norm + x[:, d:d+1])

    delta_T = 0.
    for i in range(d):
        delta_T += ps_xs[i]

    res = T_t - alpha * delta_T - f
    ps_res = [ps[i] - T_xs[i] for i in range(d)]
    return [res] + ps_res

# Distance function
dist_sphere = lambda x: 1. - torch.sum(x[:, :d] ** 2, dim=1, keepdim=True)
# Normal functions
n_gamma_sphere = NormalFunctionSphere(center=[0] * d, inner=False)

def reference_solution_pt(x):
    squared_norm = torch.sum(x[:, :d] ** 2, dim=1, keepdim=True)
    return torch.exp(0.5 * squared_norm + x[:, d:d+1])

# hard constraints
hc_gamma_sphere = HardConstraintNeumannND(
    n_gamma_sphere,
    NNWrapper(dde.nn.FNN([d+1] + 3 * [20] + [d], "tanh", "Glorot normal")),
    reference_solution_pt
)

# model
class HCNN(nn.Module):
    """
    Hard constraint model.
    """
    def __init__(self, path_prefix="") -> None:
        super(HCNN, self).__init__()
        # NNs
        self.hc_gamma = hc_gamma_sphere
        self.N_main = NNWrapper(dde.nn.FNN([d+1] + 4 * [50] + [d+1], "tanh", "Glorot normal"))
        if path_prefix != "":
            self.load(path_prefix)

    # hard constraint for each components
    def HCC_T(self, x):
        time_factor = torch.exp(-10 * x[:, d:d+1])
        return self.N_main(x, 0) * (1 - time_factor) + \
            reference_solution_pt(x) * time_factor
    
    def HCC_p_i(self, x, i):
        return self.N_main(x, i+1) * dist_sphere(x) + \
            self.hc_gamma.get_p_i(x, i)
    
    def save(self, path_prefix: str):
        torch.save(self.N_main, path_prefix + "hc_main.pth")
        self.hc_gamma.save(path_prefix, "hc_g_o.pth")
        for i in range(len(self.hc_gamma_disks)):
            self.hc_gamma_disks[i].save(path_prefix, "hc_g_d%d.pth"%i)

    def load(self, path_prefix: str):
        self.N_main = torch.load(path_prefix + "hc_main.pth")
        self.hc_gamma.load(path_prefix, "hc_g_o.pth")
        for i in range(len(self.hc_gamma_disks)):
            self.hc_gamma_disks[i].load(path_prefix, "hc_g_d%d.pth"%i)

    def forward(self, x) -> torch.Tensor:
        self.N_main.clear_res()
        self.hc_gamma.clear_res()
        return torch.cat(
            [self.HCC_T(x)] + [
                self.HCC_p_i(x, i) for i in range(d)
            ], dim=1)
