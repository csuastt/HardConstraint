import deepxde.losses as losses_module
from deepxde.callbacks import Callback
import copy
import numpy as np
import torch


class PINNTester(Callback):

    def __init__(self, data_path, dim, test_fn):
        super().__init__()
        self.data_path = data_path
        self.dim = dim
        self.test_fn = test_fn
        self.m_abs_e_res = []
        self.m_r_abs_e_res = []
        self.m_loss_res = []

    def on_epoch_end(self):
        net = self.model.net
        test_res = self.test_fn(self.data_path, self.dim, net)
        self.m_abs_e_res.append(test_res[0][0])
        self.m_r_abs_e_res.append(test_res[1][0])
        self.m_loss_res.append(np.sum(self.model.train_state.loss_train))


class PINNGradientTracker(Callback):

    def __init__(self, num_pdes):
        super().__init__()
        self.num_pdes = num_pdes
        self.m_gradients = []
        self.conds = []
        self.last_loss = None
        self.last_params = None
        self.steps = []
        self.loss_fn = losses_module.get("MSE")

    def on_epoch_end(self):
        model = self.model.net
        params = []
        for param in model.parameters():
            params.append(param.reshape(-1))
        params = torch.cat(params)
        # get the loss and parameters
        outputs = self.model.net(self.model.net.inputs.float())
        losses = self.model.data.losses(None, outputs, self.loss_fn, self.model)
        # find mean|\nabla_{\theta}L_r|
        losses_r = torch.sum(torch.stack(losses[:self.num_pdes]))
        m_grad_r = [] 
        for param in self.model.net.parameters():
            grads = torch.autograd.grad(losses_r, param, retain_graph=True, allow_unused=True)
            if grads[0] is not None:
                m_grad_r.append(torch.abs(grads[0]).reshape(-1))
            else:
                m_grad_r.append(torch.zeros_like(param))
        self.m_gradients.append(torch.mean(torch.cat(m_grad_r)).item())
        self.steps.append(self.model.train_state.epoch)
        loss = np.sum(self.model.train_state.loss_train[:self.num_pdes])
        loss = np.sum(loss)
        # calculate cond
        if self.last_params is None:
            # the first epoch
            self.conds.append(None)
        else:
            self.conds.append(
                np.abs(loss - self.last_loss) / torch.norm(params-self.last_params).item()
            )
        self.last_params = params
        self.last_loss = loss


class PINNLRAdaptor(Callback):
    """
    PINN callback for learning rate annealing algorithm of physics-informed neural networks.
    """

    def __init__(self, loss_weight, num_pdes, alpha, mode="max"):
        '''
        loss_weight - initial loss weights\n
        num_pdes - the number of the PDEs (boundary conditions excluded)\n
        alpha - parameter of moving average\n
        mode - "max" (PINN-LA), "mean" (PINN-LA-2)
        '''
        super().__init__()
        self.loss_weight = loss_weight
        self.num_pdes = num_pdes
        self.alpha = alpha
        self.loss_fn = losses_module.get("MSE")
        self.mode = mode

    def on_epoch_end(self):
        # get the loss and parameters
        outputs = self.model.net(self.model.net.inputs.float())
        losses = self.model.data.losses(None, outputs, self.loss_fn, self.model)
        # find max|\nabla_{\theta}L_r|
        losses_r = torch.sum(torch.stack(losses[:self.num_pdes]))
        m_grad_r = [] 
        for param in self.model.net.parameters():
            grads = torch.autograd.grad(losses_r, param, retain_graph=True, allow_unused=True)
            if grads[0] is not None:
                m_grad_r.append(torch.abs(grads[0]).reshape(-1))
            else:
                m_grad_r.append(torch.zeros_like(param))
        if self.mode == "mean":
            m_grad_r = torch.mean(torch.cat(m_grad_r)).item()
        else:
            m_grad_r = torch.max(torch.cat(m_grad_r)).item()
        # adapt the weights for each bc term
        for i in range(self.num_pdes, len(self.loss_weight)):
            grads_bc = []
            for param in self.model.net.parameters():
                grads = torch.autograd.grad(losses[i], param, retain_graph=True, allow_unused=True)
                if grads[0] is not None:
                    grads_bc.append(torch.abs(grads[0]).reshape(-1))
                else:
                    grads_bc.append(torch.zeros_like(param))
            lambda_hat = m_grad_r / (torch.mean(torch.cat(grads_bc)).item() * self.loss_weight[i])
            self.loss_weight[i] = (1 - self.alpha) * self.loss_weight[i] + self.alpha * lambda_hat


class PINNLRScheduler(Callback):
    """
    PINN callback for learning rate scheduler.
    """
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_epoch_end(self):
        # get the loss and parameters
        losses = np.array(self.model.train_state.loss_train)
        m_loss = np.mean(losses)
        self.scheduler.step(m_loss)


class PINNModelSaver(Callback):
    """
    PINN callback for saving the weights of physics-informed neural networks.
    """

    def __init__(self, init_weights=None):
        super().__init__()
        # if the model got nan outputs
        self.got_nan = False
        # the weights of the last epoch
        if init_weights is not None:
            self.weights = copy.deepcopy(init_weights)

    def on_epoch_end(self):
        # get the loss and parameters
        losses = np.array(self.model.train_state.loss_train)
        if np.isnan(losses).any():
            # do not save when it has nan outputs
            self.got_nan = True
        else:
            self.weights = copy.deepcopy(self.model.net.state_dict())