import torch
from torch import optim
import numpy as np
import deepxde as dde
from deepxde import gradients as dde_grad
from src.FBPINN.fbpinn import FBPINN
from src.PFNN.pfnn import PFNN
from src.configs.case1.xpinn import pde_xpinn
from src.utils.no_stdout_context import nostdout
from src.utils.utils import plot_lines, test_time, Tester
from src.utils.pinn_callback import PINNLRAdaptor, PINNLRScheduler, PINNModelSaver, PINNTester
from src.configs.case1.params import model_path_prefix, data_path, H, L
from src.configs.case1.pinn import spatial_time_domain, time_domain, pde_pinn, ic_bcs, num_bcs, num_pdes, lr_alpha
from src.configs.case1.hc import pde_hc, HCNN
from src.configs.case1.fbpinn import sigma
from src.configs.case1.pfnn import tot_points, loss_pfnn, initial_condition
from src.xPINN.interface_conditions import Subdomains
from src.xPINN.xPINN import xPINN


# Model Selection (default: HC)
TEST_PINN = False
TEST_FBPINN = False # this choice is valid if TEST_PINN == True
PINN_LR_ANNEALING = False # this choice is valid if TEST_PINN == True
PINN_LR_ANNEALING_2 = False # this choice is valid if TEST_PINN == True
TEST_XPINN = False
TEST_PFNN = False
TEST_PFNN_2 = False # this choice is valid if TEST_PFNN == True

# Other Configurations
LOAD_MODEL = False
SAVE_MODEL = False
TEST_WHILE_TRAIN = False


def train_pinn():
    n_epochs = 5000
    lr = 0.01
    data = dde.data.TimePDE(
        spatial_time_domain,
        pde_pinn,
        ic_bcs,
        num_domain=8192,
        num_boundary=512,
        num_initial=512
    )
    if TEST_FBPINN:
        net = torch.nn.DataParallel(FBPINN(sigma, L, H, 4, 6, 3, 3, 30, 1))
    else:
        net = dde.nn.FNN([3] + 4 * [50] + [1], "tanh", "Glorot normal")
    if LOAD_MODEL:
        net = torch.load(model_path_prefix + "pinn.pth")
    # train
    optimizer = optim.Adam(
        net.parameters(),
        lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.75, min_lr=1e-5
    )
    test_callback = PINNTester(data_path, spatial_time_domain.dim, Tester.test_while_train)
    loss_weights = [1.] * (num_pdes + num_bcs)
    lr_adaptor_callback = PINNLRAdaptor(
        loss_weights, num_pdes, lr_alpha,
        mode="max" if PINN_LR_ANNEALING else "mean"
    )
    lr_scheduler_callback = PINNLRScheduler(scheduler)
    resampler = dde.callbacks.PDEResidualResampler(period=10)
    callbacks = [lr_scheduler_callback, resampler]
    if TEST_WHILE_TRAIN:
        callbacks.append(test_callback)
    if PINN_LR_ANNEALING or PINN_LR_ANNEALING_2:
        callbacks.append(lr_adaptor_callback)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    with nostdout():
        model.train(epochs=n_epochs, callbacks=callbacks, display_every=1)
    # lbfgs
    model_saver = PINNModelSaver()
    resampler = dde.callbacks.PDEResidualResampler(period=1)
    model.compile("L-BFGS", loss_weights=loss_weights)
    model.train(callbacks=[resampler, model_saver])
    if model_saver.got_nan:
        model.net.load_state_dict(model_saver.weights)
    m_loss_res = test_callback.m_loss_res
    m_abs_e_res = test_callback.m_abs_e_res
    m_r_abs_e_res = test_callback.m_r_abs_e_res
    net = model.net
    if TEST_FBPINN:
        net = net.module
    # save the model
    if SAVE_MODEL:
        torch.save(net, model_path_prefix + "pinn.pth")
    return net, m_loss_res, m_abs_e_res, m_r_abs_e_res


def train_hc():
    n_epochs = 5000
    lr = 0.01
    net = HCNN(model_path_prefix if LOAD_MODEL else "")
    optimizer = optim.Adam(
        net.parameters(),
        lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.75, min_lr=1e-5
    )
    # prepare data set
    X = torch.tensor(spatial_time_domain.random_points(8192)).float()
    X.requires_grad = True
    # train
    m_abs_e_res, m_r_abs_e_res, m_loss_res = [], [], []
    print("Training...")
    for i in range(n_epochs):
        pred = net(X)
        loss = torch.cat(pde_hc(X, pred), dim=1) 
        loss = torch.sum(loss ** 2, dim=1)
        loss = torch.mean(loss)
        # Backpropagation
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        # update the lr
        loss_val = loss.item()
        scheduler.step(loss_val)
        m_loss_res.append(loss_val)
        # test while train
        if TEST_WHILE_TRAIN:
            # need not net.eval() right now
            test_res = Tester.test_while_train(data_path, spatial_time_domain.dim, net)
            m_abs_e_res.append(test_res[0][0])
            m_r_abs_e_res.append(test_res[1][0])
        if (i+1) % 100 == 0 or i == 0:
            print(f"[Epoch {i+1}/{n_epochs}] loss: {loss_val:>7f}")
            # clear the cache of the grads
            dde_grad.clear()
        if i % 10 == 0:
            X = torch.tensor(spatial_time_domain.random_points(8192)).float()
            X.requires_grad = True
    # l-bfgs
    resampler = dde.callbacks.PDEResidualResampler(period=1)
    data = dde.data.TimePDE(
        spatial_time_domain,
        pde_hc,
        [],
        num_domain=8192
    )
    model = dde.Model(data, net)
    model.compile("L-BFGS")
    model.train(callbacks=[resampler])
    print("Finish training!")
    # save the model
    if SAVE_MODEL:
        net.save(model_path_prefix)
    return net, m_loss_res, m_abs_e_res, m_r_abs_e_res


def train_xpinn():
    # prepare for xPINN
    subdomains = Subdomains(L, H, 4, 6, 1, spatial_time_domain, temporal_domain=time_domain)
    interface_points = subdomains.generate_interface_points(512)
    interface_conditions = subdomains.generate_interface_conditions()
    # set data
    n_epochs = 5000
    lr = 0.01
    data = dde.data.TimePDE(
        spatial_time_domain,
        pde_xpinn,
        ic_bcs + interface_conditions,
        num_domain=8192,
        num_boundary=512,
        num_initial=512,
        anchors=interface_points
    )
    net = torch.nn.DataParallel(xPINN(L, H, 4, 6, 3, 3, 30, 1, pde_pinn))
    if LOAD_MODEL:
        net = torch.load(model_path_prefix + "xpinn.pth")
    # train
    optimizer = optim.Adam(
        net.parameters(),
        lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.75, min_lr=1e-5
    )
    test_callback = PINNTester(data_path, spatial_time_domain.dim, Tester.test_while_train)
    lr_scheduler_callback = PINNLRScheduler(scheduler)
    resampler = dde.callbacks.PDEResidualResampler(period=10)
    callbacks = [lr_scheduler_callback, resampler]
    if TEST_WHILE_TRAIN:
        callbacks.append(test_callback)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    with nostdout():
        model.train(epochs=n_epochs, callbacks=callbacks, display_every=1)
    # lbfgs
    model_saver = PINNModelSaver()
    resampler = dde.callbacks.PDEResidualResampler(period=1)
    model.compile("L-BFGS")
    model.train(callbacks=[resampler, model_saver])
    if model_saver.got_nan:
        model.net.load_state_dict(model_saver.weights)
    m_loss_res = test_callback.m_loss_res
    m_abs_e_res = test_callback.m_abs_e_res
    m_r_abs_e_res = test_callback.m_r_abs_e_res
    net = model.net.module
    # save the model
    if SAVE_MODEL:
        torch.save(net, model_path_prefix + "xpinn.pth")
    # set model to evaluation mode
    net.set_eval()
    return net, m_loss_res, m_abs_e_res, m_r_abs_e_res


def train_pfnn():
    n_epochs = 5000
    lr = 0.01
    data = dde.data.TimePDE(
        spatial_time_domain,
        loss_pfnn,
        [],
        anchors=tot_points
    )
    net = PFNN(
        initial_condition, 3, 1,
        ddpinn=torch.nn.DataParallel(FBPINN(sigma, L, H, 4, 6, 3, 3, 30, 1)) if TEST_PFNN_2 else None
    )
    if LOAD_MODEL:
        net = torch.load(model_path_prefix + "pfnn.pth")
    # train
    optimizer = optim.Adam(
        net.parameters(),
        lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.75, min_lr=1e-5
    )
    test_callback = PINNTester(data_path, spatial_time_domain.dim, Tester.test_while_train)
    lr_scheduler_callback = PINNLRScheduler(scheduler)
    resampler = dde.callbacks.PDEResidualResampler(period=10)
    callbacks = [lr_scheduler_callback, resampler]
    if TEST_WHILE_TRAIN:
        callbacks.append(test_callback)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    with nostdout():
        model.train(epochs=n_epochs, callbacks=callbacks, display_every=1)
    # lbfgs
    model_saver = PINNModelSaver()
    model.compile("L-BFGS")
    model.train(callbacks=[model_saver])
    if model_saver.got_nan:
        model.net.load_state_dict(model_saver.weights)
    m_loss_res = test_callback.m_loss_res
    m_abs_e_res = test_callback.m_abs_e_res
    m_r_abs_e_res = test_callback.m_r_abs_e_res
    net = model.net
    if TEST_PFNN_2:
        net.net = net.net.module
    # save the model
    if SAVE_MODEL:
        torch.save(net, model_path_prefix + "pfnn.pth")
    return net, m_loss_res, m_abs_e_res, m_r_abs_e_res


if __name__ == "__main__":
    if TEST_PINN:
        net, m_loss_res, m_abs_e_res, m_r_abs_e_res = train_pinn()
    elif TEST_XPINN:
        net, m_loss_res, m_abs_e_res, m_r_abs_e_res = train_xpinn()
    elif TEST_PFNN:
        net, m_loss_res, m_abs_e_res, m_r_abs_e_res = train_pfnn()
    else:
        net, m_loss_res, m_abs_e_res, m_r_abs_e_res = train_hc()

    # test the model
    # net.eval()
    test_time(data_path, spatial_time_domain.dim - 1, net)
    # plot lines of testing res while training
    if TEST_WHILE_TRAIN:
        plot_lines(
            [list(range(1, len(m_abs_e_res) + 1)), 
                np.array(m_loss_res) / np.max(m_loss_res), 
                np.array(m_abs_e_res) / np.max(m_abs_e_res), 
                np.array(m_r_abs_e_res) / np.max(m_r_abs_e_res), 
            ],
            "Epochs",
            "Normalized result",
            ["loss", "m_abs_e", "m_r_abs_e"], "outs/e_while_training.png",
            is_log=True
        )
