import deepxde as dde
from src.configs.case1.params import *

def pde_xpinn(x, T):
    T_t = dde.grad.jacobian(T, x, i=0, j=2)
    T_xx = dde.grad.hessian(T, x, i=0, j=0, component=0)
    T_yy = dde.grad.hessian(T, x, i=1, j=1, component=0)
    return T_t - k * (T_xx + T_yy)