import deepxde as dde
import numpy as np
from src.configs.case2.params import *
from src.utils.pinn_bc import NormalBC


rec = dde.geometry.Rectangle(xmin=xmin, xmax=xmax)
airfoil = dde.geometry.Polygon(anchor_points)
spatial_domain = dde.geometry.CSGDifference(rec, airfoil)


def pde_pinn(x, u):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (
        u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy)
    )
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]

# u0, v0
in_u_bc = dde.icbc.DirichletBC(
    spatial_domain, lambda _: 1., 
    lambda x, on_bc: on_bc and (np.isclose(x[0], xmin[0]) or 
        np.isclose(x[1], xmin[1]) or np.isclose(x[1], xmax[1])), 
    component=0
)

in_v_bc = dde.icbc.DirichletBC(
    spatial_domain, lambda _: 0., 
    lambda x, on_bc: on_bc and (np.isclose(x[0], xmin[0]) or 
        np.isclose(x[1], xmin[1]) or np.isclose(x[1], xmax[1])), 
    component=1
)

out_p_bc = dde.icbc.DirichletBC(
    spatial_domain, lambda _: 1., 
    lambda x, on_bc: on_bc and np.isclose(x[0], xmax[0]), 
    component=2
)

airfoil_bc = NormalBC(
    spatial_domain, lambda _: 0.,
    lambda x, on_bc: on_bc and (not rec.on_boundary(x)),
    component=0
)

ic_bcs = [in_u_bc, in_v_bc, out_p_bc, airfoil_bc]

# parameters for PINN
# moving average in learning rate annealing
lr_alpha = 0.1
# num pdes
num_pdes = 3
num_bcs = len(ic_bcs)