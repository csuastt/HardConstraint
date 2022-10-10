import torch
from torch import nn


# MLP with residual connection
class ResNet(nn.Module):
    def __init__(self, num_res_layers, input_dim, hidden_dim, output_dim):
        '''
        num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        device: which device to use
        '''

        super(ResNet, self).__init__()
        self.num_layers = num_res_layers
        # Multi-layer model
        self.linears = torch.nn.ModuleList()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        for _ in range(num_res_layers):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_loc, x_bou=None):
        if x_bou is None:
            x = x_loc
        else:
            x = torch.cat((x_bou, x_loc), 1)
        x = torch.tanh(self.fc1(x))
        last_t = x
        res_connect = False
        for layer in range(self.num_layers):
            if res_connect:
                x = torch.tanh(self.linears[layer](x)) + last_t
                last_t = x
            else:
                x = torch.tanh(self.linears[layer](x))
            res_connect = not res_connect
        x = self.fc2(x)
        return x
