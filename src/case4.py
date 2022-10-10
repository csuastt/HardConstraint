import deepxde as dde
import numpy as np
import torch
from src.utils.no_stdout_context import nostdout
from src.utils.pinn_callback import PINNGradientTracker
from src.utils.utils import plot_lines


# Problem Selection (default: Poisson equation)
TEST_SH = False # test on the nonlinear Schrödinger equation 

# Other Configuration
a = 2

# DO NOT MODIFY THIS VARIABLE
EXTRA_FIELDS = False 


def train_poisson():
    def pde(x, y):
        if EXTRA_FIELDS:
            p = y[:, 1:]
            du_x = dde.grad.jacobian(y, x, i=0, j=0)
            dp_x = dde.grad.jacobian(y, x, i=1, j=0)
            return [
                dp_x + a ** 2 * torch.sin(a * x),
                p - du_x
            ]
        dy_xx = dde.grad.hessian(y, x)
        return dy_xx + a ** 2 * torch.sin(a * x)

    def boundary(_, on_boundary):
        return on_boundary

    def y_exact(x):
        return np.sin(a * x)
    # training
    geom = dde.geometry.Interval(0, 2 * np.pi)
    bc = dde.icbc.DirichletBC(geom, y_exact, boundary)
    data = dde.data.PDE(geom, pde, bc, 128, 2)

    if EXTRA_FIELDS:
        layer_size = [1] + [50] * 3 + [2]
    else:
        layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    callback = PINNGradientTracker(num_pdes=2 if EXTRA_FIELDS else 1) 
    model.train(epochs=10000, display_every=1, callbacks=[callback])
    m_gradients = np.array(callback.m_gradients)
    steps = np.array(callback.steps)
    conds = np.array(callback.conds)
    return steps, m_gradients, conds


def train_schrodinger():
    '''
    Source: anonymous
    '''
    x_lower = -5
    x_upper = 5
    t_lower = 0
    t_upper = np.pi / 2
    def pde(x, y):
        """
        INPUTS:
            x: x[:,0] is x-coordinate
            x[:,1] is t-coordinate
            y: Network output, in this case:
                y[:,0] is u(x,t) the real part
                y[:,1] is v(x,t) the imaginary part
        OUTPUT:
            The pde in standard form i.e. something that must be zero
        """
        if EXTRA_FIELDS:
            u = y[:, 0:1]
            v = y[:, 1:2]
            _u_x = y[:, 2:3]
            _v_x = y[:, 3:4]

            # In 'jacobian', i is the output component and j is the input component
            u_t = dde.grad.jacobian(y, x, i=0, j=1)
            v_t = dde.grad.jacobian(y, x, i=1, j=1)

            u_x = dde.grad.jacobian(y, x, i=0, j=0)
            v_x = dde.grad.jacobian(y, x, i=1, j=0)

            # In 'hessian', i and j are both input components. (The Hessian could be in principle something like d^2y/dxdt, d^2y/d^2x etc)
            # The output component is selected by "component"
            u_xx = dde.grad.jacobian(y, x, i=2, j=0)
            v_xx = dde.grad.jacobian(y, x, i=3, j=0)

            f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
            f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

            return [f_u, f_v, u_x - _u_x, v_x - _v_x]
        u = y[:, 0:1]
        v = y[:, 1:2]

        # In 'jacobian', i is the output component and j is the input component
        u_t = dde.grad.jacobian(y, x, i=0, j=1)
        v_t = dde.grad.jacobian(y, x, i=1, j=1)

        u_x = dde.grad.jacobian(y, x, i=0, j=0)
        v_x = dde.grad.jacobian(y, x, i=1, j=0)

        # In 'hessian', i and j are both input components. (The Hessian could be in principle something like d^2y/dxdt, d^2y/d^2x etc)
        # The output component is selected by "component"
        u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)

        f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

        return [f_u, f_v]
    
    # Space and time domains/geometry (for the deepxde model)
    space_domain = dde.geometry.Interval(x_lower, x_upper)
    time_domain = dde.geometry.TimeDomain(t_lower, t_upper)
    geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

    # Boundary and Initial conditions
    # Periodic Boundary conditions
    bc_u_0 = dde.icbc.PeriodicBC(
        geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0
    )
    bc_u_1 = dde.icbc.PeriodicBC(
        geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0
    )
    bc_v_0 = dde.icbc.PeriodicBC(
        geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1
    )
    bc_v_1 = dde.icbc.PeriodicBC(
        geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=1
    )

    # Initial conditions
    def init_cond_u(x):
        "2 sech(x)"
        return 2 / np.cosh(x[:, 0:1])


    def init_cond_v(x):
        return 0


    ic_u = dde.icbc.IC(geomtime, init_cond_u, lambda _, on_initial: on_initial, component=0)
    ic_v = dde.icbc.IC(geomtime, init_cond_v, lambda _, on_initial: on_initial, component=1)

    # training
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_u_0, bc_u_1, bc_v_0, bc_v_1, ic_u, ic_v],
        num_domain=1000,
        num_boundary=20,
        num_initial=200,
        train_distribution="pseudo",
    )

    if EXTRA_FIELDS:
        layer_size = [2] + [100] * 5 + [4]
    else:
        layer_size = [2] + [100] * 5 + [2]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    callback = PINNGradientTracker(num_pdes=4 if EXTRA_FIELDS else 2) 
    model.train(epochs=10000, display_every=1, callbacks=[callback])
    m_gradients = np.array(callback.m_gradients)
    steps = np.array(callback.steps)
    conds = np.array(callback.conds)
    return steps, m_gradients, conds


if __name__ == "__main__":
    with nostdout():
        steps, m_gradients_raw, conds_raw = train_schrodinger() if TEST_SH else train_poisson()
        EXTRA_FIELDS = True
        steps, m_gradients_ef, conds_ef = train_schrodinger() if TEST_SH else train_poisson()
    # plot the gradients history
    plot_lines(
        [steps[1::100], np.abs(m_gradients_raw[1::100]), np.abs(m_gradients_ef[1::100])],
        "Steps",
        "Mean absolute gradients (abs)",
        ["Origin", "Extra Fields"], "outs/mean_gradients_while_training.png",
        is_log=True
    )
    # plot the cond history
    plot_lines(
        [steps[1::100], np.abs(conds_raw[1::100]), np.abs(conds_ef[1::100])],
        "Steps",
        "Condition numbers (abs)",
        ["Origin", "Extra Fields"], "outs/conds_while_training.png",
        is_log=True
    )
