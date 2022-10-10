import torch
from torch import nn
from src.utils.resnet import ResNet
import deepxde as dde


class FBPINN(nn.Module):
    """
    Doamin decomposition PINN with window functions.
    ## Parameter
    sigma - scale factor\n
    L - length\n
    H - height\n
    n_col - the number of columns of the sub-domains\n
    n_row - the number of rows of the sub-domains\n
    num_layers, input_dim, hidden_dim, output_dim - the parameters of each sub-network\n
    xmin - the offset of the left-bottom point
    """
    def __init__(
        self, sigma, L, H, n_col, n_row,
        num_layers, input_dim, hidden_dim, output_dim,
        is_res_net=False, xmin=None
    ) -> None:
        super(FBPINN, self).__init__()
        self.sigma = sigma
        self.L = L
        self.H = H
        if xmin is None:
            self.xmin = [- L / 2, - H / 2]
        else:
            self.xmin = xmin
        self.n_col = n_col
        self.n_row = n_row
        if is_res_net:
            nets = [ResNet(num_layers, input_dim, hidden_dim, output_dim) for _ in range(n_col * n_row)]
        else:
            nets = [dde.nn.FNN([input_dim] + num_layers * [hidden_dim] + [output_dim], "tanh", "Glorot normal") for _ in range(n_col * n_row)]
        self.nets = nn.ModuleList(nets)
        self.lower_bs = None

    def forward(self, x) -> torch.Tensor:
        if self.lower_bs is None:
            self.lower_bs = [
                [
                    torch.tensor([self.L * j / self.n_col + self.xmin[0],
                        self.H * i / self.n_row + self.xmin[1]])
                    for j in range(self.n_col)
                ] for i in range(self.n_row)
            ]
            self.upper_bs = [
                [
                    torch.tensor([self.L * (j+1) / self.n_col + self.xmin[0],
                        self.H * (i+1) / self.n_row  + self.xmin[1]])
                    for j in range(self.n_col)
                ] for i in range(self.n_row)
            ]
            self.centers = [
                [
                    torch.tensor([(self.L * j / self.n_col + self.L * (j+1) / self.n_col) / 2 + self.xmin[0],
                    (self.H * i / self.n_row + self.H * (i+1) / self.n_row) / 2 + self.xmin[1]])
                    for j in range(self.n_col)
                ] for i in range(self.n_row)
            ]
            self.subdomain_size = torch.tensor([self.L / self.n_col, self.H / self.n_row])
        res = None
        spatial_x = x[:, :2]
        for i in range(self.n_row):
            for j in range(self.n_col):
                window_res = torch.sigmoid(
                    (spatial_x[:, 0:1] - self.lower_bs[i][j][0]) / self.sigma
                ) * torch.sigmoid(
                    (self.upper_bs[i][j][0:1] - spatial_x[:, 0:1]) / self.sigma
                ) * torch.sigmoid(
                    (spatial_x[:, 1:] - self.lower_bs[i][j][1]) / self.sigma
                ) * torch.sigmoid(
                    (self.upper_bs[i][j][1] - spatial_x[:, 1:]) / self.sigma
                )
                # normalization
                spatial_x_normalized = (spatial_x - self.centers[i][j]) / \
                    self.subdomain_size
                # recover temporal dimension
                if x.shape[1] > 2:
                    x_normalized = torch.cat([spatial_x_normalized, x[:, 2:3]], dim=1)
                else:
                    x_normalized = spatial_x_normalized
                nn_res = self.nets[i * self.n_col + j](x_normalized)
                cal_res = window_res.expand(-1, nn_res.shape[1]) * nn_res
                if res is None:
                    res = cal_res
                else:
                    res += cal_res
        return res
