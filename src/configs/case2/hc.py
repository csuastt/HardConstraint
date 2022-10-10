import torch
from torch import nn
import deepxde as dde
from src.HC.hard_constraint import HardConstraintNeumann2D
from src.HC.hard_constraint_collector import HardConstraintCollector
from src.HC.l_functions import LFunctionAxisLine, LFunctionOpenRectangle, LFunctionPolygon
from src.HC.normal_function import NormalFunctionPolygon
from src.configs.case2.pinn import spatial_domain, rec, airfoil
from src.configs.case2.params import *
from src.utils.nn_wrapper import NNWrapper


def pde_hc(x, u):
    u_vel, v_vel, p, _u_x, _u_y, _v_x, _v_y  = \
        u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4], u[:, 4:5], u[:, 5:6], u[:, 6:7]
    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    u_vel_xx = dde.grad.jacobian(u, x, i=3, j=0)
    u_vel_yy = dde.grad.jacobian(u, x, i=4, j=1)
    v_vel_xx = dde.grad.jacobian(u, x, i=5, j=0)
    v_vel_yy = dde.grad.jacobian(u, x, i=6, j=1)

    res_u_x = (_u_x - u_vel_x)
    res_u_y = (_u_y - u_vel_y)
    res_v_x = (_v_x - v_vel_x)
    res_v_y = (_v_y - v_vel_y)
    momentum_x = (
        u_vel * _u_x + v_vel * _u_y + p_x - nu * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        u_vel * _v_x + v_vel * _v_y + p_y - nu * (v_vel_xx + v_vel_yy)
    )
    continuity = _u_x + _v_y

    return [res_u_x, res_u_y, res_v_x, res_v_y, momentum_x, momentum_y, continuity]

# Helper functions (input: x, a B x d tensor)
X_rec_bc = rec.random_boundary_points(512)
X_airfoil_bc = airfoil.random_boundary_points(512)
# M function
beta = 4.0
M = lambda x: torch.logsumexp(-beta * x, dim=1, keepdim=True) / (-beta)
# Lambda functions
l_gamma_openrec = LFunctionOpenRectangle(X_airfoil_bc, rec, m_function=M)
l_gamma_right = LFunctionAxisLine(X_airfoil_bc, rec, xmax[0], 0, is_left=False)
l_gamma_airfoil = LFunctionPolygon(X_rec_bc, airfoil, spatial_domain)
# Normal functions
n_gamma_airfoil = NormalFunctionPolygon(airfoil)

# hard constraints
hc_gamma_airfoil = HardConstraintNeumann2D(
    n_gamma_airfoil, 
    NNWrapper(dde.nn.FNN([2] + 4 * [40] + [1], "tanh", "Glorot normal")),
    lambda _: 0.
)

# model
class HCNN(nn.Module):
    """
    Hard constraint model.
    """
    def __init__(self, path_prefix="") -> None:
        super(HCNN, self).__init__()
        # NNs
        self.hc_gamma_airfoil = hc_gamma_airfoil
        self.N_main = NNWrapper(dde.nn.FNN([2] + 6 * [50] + [7], "tanh", "Glorot normal"))
        if path_prefix != "":
            self.load(path_prefix)
        # hard constraint for each components
        self.HCC_u = HardConstraintCollector(
            0, M, [l_gamma_openrec, l_gamma_airfoil],
            [lambda _: 1., hc_gamma_airfoil.get_p_1], self.N_main
        )
        self.HCC_v = HardConstraintCollector(
            1, M, [l_gamma_openrec, l_gamma_airfoil],
            [lambda _: 0., hc_gamma_airfoil.get_p_2], self.N_main
        )
        self.HCC_p = HardConstraintCollector(
            2, M, [l_gamma_right],
            [lambda _: 1.], self.N_main
        )
    
    def save(self, path_prefix: str):
        torch.save(self.N_main, path_prefix + "hc_main.pth")
        self.hc_gamma_airfoil.save(path_prefix, "hc_g_a.pth")

    def load(self, path_prefix: str):
        self.N_main = torch.load(path_prefix + "hc_main.pth")
        self.hc_gamma_airfoil.load(path_prefix, "hc_g_a.pth")

    def forward(self, x) -> torch.Tensor:
        self.N_main.clear_res()
        self.hc_gamma_airfoil.clear_res()
        return torch.cat(
            [
            self.HCC_u(x),
            self.HCC_v(x),
            self.HCC_p(x)
            ] + [self.N_main(x, i) for i in range(3, 7)]
            , dim=1)
