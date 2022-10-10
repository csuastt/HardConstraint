import torch
from torch import optim
import numpy as np
import deepxde as dde
from deepxde import gradients as dde_grad
from src.PFNN.pfnn import PFNN
from src.configs.case3.hc import HCNN, pde_hc
from src.utils.no_stdout_context import nostdout
from src.utils.utils import test_time_with_reference_solution
from src.utils.pinn_callback import PINNLRAdaptor, PINNLRScheduler, PINNModelSaver
from src.configs.case3.params import model_path_prefix, d
from src.configs.case3.pinn import reference_solution, spatial_time_domain, \
    pde_pinn, ic_bcs, num_bcs, num_pdes, lr_alpha, spatial_domain
from src.configs.case3.pfnn import loss_pfnn, reference_solution_pt, tot_points


# Model Selection (default: HC)
TEST_PINN = False
TEST_PFNN = False
PINN_LR_ANNEALING = False # this choice is valid if TEST_PINN == True
PINN_LR_ANNEALING_2 = False # this choice is valid if TEST_PINN == True

# Other Configurations
LOAD_MODEL = False
SAVE_MODEL = False


def train_pinn():
    n_epochs = 5000
    lr = 0.01
    data = dde.data.TimePDE(
        spatial_time_domain,
        pde_pinn,
        ic_bcs,
        num_domain=1000,
        num_boundary=100,
        num_initial=100
    )
    net = dde.nn.FNN([d + 1] + 4 * [50] + [1], "tanh", "Glorot normal")
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
    loss_weights = [1.] * (num_pdes + num_bcs)
    lr_adaptor_callback = PINNLRAdaptor(
        loss_weights, num_pdes, lr_alpha,
        mode="max" if PINN_LR_ANNEALING else "mean"
    )
    lr_scheduler_callback = PINNLRScheduler(scheduler)
    resampler = dde.callbacks.PDEResidualResampler(period=10)
    callbacks = [lr_scheduler_callback, resampler]
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
    net = model.net
    # save the model
    if SAVE_MODEL:
        torch.save(net, model_path_prefix + "pinn.pth")
    return net


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
    X = torch.tensor(spatial_time_domain.random_points(1000)).float()
    X.requires_grad = True
    # train
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
        if (i+1) % 100 == 0 or i == 0:
            print(f"[Epoch {i+1}/{n_epochs}] loss: {loss_val:>7f}")
            # clear the cache of the grads
            dde_grad.clear()
        if i % 10 == 0:
            X = torch.tensor(spatial_time_domain.random_points(1000)).float()
            X.requires_grad = True
    # l-bfgs
    resampler = dde.callbacks.PDEResidualResampler(period=1)
    model_saver = PINNModelSaver(net.state_dict())
    data = dde.data.TimePDE(
        spatial_time_domain,
        pde_hc,
        [],
        num_domain=1000
    )
    model = dde.Model(data, net)
    model.compile("L-BFGS")
    model.train(callbacks=[resampler, model_saver])
    if model_saver.got_nan:
        net.load_state_dict(model_saver.weights)
    print("Finish training!")
    # save the model
    if SAVE_MODEL:
        net.save(model_path_prefix)
    return net


def train_pfnn():
    n_epochs = 5000
    lr = 0.01
    data = dde.data.TimePDE(
        spatial_time_domain,
        loss_pfnn,
        [],
        anchors=tot_points
    )
    net = PFNN(reference_solution_pt, d + 1, 1)
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
    lr_scheduler_callback = PINNLRScheduler(scheduler)
    resampler = dde.callbacks.PDEResidualResampler(period=10)
    callbacks = [lr_scheduler_callback, resampler]
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
    net = model.net
    # save the model
    if SAVE_MODEL:
        torch.save(net, model_path_prefix + "pfnn.pth")
    return net


if __name__ == "__main__":
    if TEST_PINN:
        net = train_pinn()
    elif TEST_PFNN:
        net = train_pfnn()
    else:
        net = train_hc()

    # test the model
    # net.eval()
    test_X = spatial_domain.random_points(1000)
    test_X = np.concatenate([
        np.concatenate((test_X, np.ones((test_X.shape[0], 1)) * t / 10), axis=1)
        for t in range(11)
    ], axis=0)
    test_time_with_reference_solution(reference_solution, test_X, net)
