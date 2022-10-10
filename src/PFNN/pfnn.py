import torch
from torch import nn
import deepxde as dde

class PFNN(nn.Module):
    """
    Penalty-Free Neural Network for time-dependent PDEs.
    """
    def __init__(self, ic_fn, num_input, num_output, ddpinn=None) -> None:
        '''
        ddpinn - a domain-decomposition based PINN. Specify this nn to select the PFNN-2.
        '''
        super(PFNN, self).__init__()
        self.ic_fn = ic_fn
        if ddpinn is None:
            self.net = dde.nn.FNN([num_input] + 4 * [50] + [num_output], 
                "tanh", "Glorot normal")
        else:
            self.net = ddpinn

    def forward(self, x) -> torch.Tensor:
        # -1 is the time dimension
        return self.net(x) * x[:, -1:] + self.ic_fn(x)