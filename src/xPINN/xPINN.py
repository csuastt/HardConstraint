import torch
from torch import nn
import deepxde as dde


class xPINN(nn.Module):
    """
    Doamin decomposition PINN in a hard way.
    ## Parameter
    L - length\n
    H - height\n
    n_col - the number of columns of the sub-domains\n
    n_row - the number of rows of the sub-domains\n
    num_layers, input_dim, hidden_dim, output_dim - the parameters of each sub-network\n
    pde_fn - function to calculate pde residuals\n
    xmin - the offset of the left-bottom point
    """
    def __init__(
        self, L, H, n_col, n_row,
        num_layers, input_dim, hidden_dim, output_dim, pde_fn,
        xmin=None
    ) -> None:
        super(xPINN, self).__init__()
        self.L = L
        self.H = H
        self.n_col = n_col
        self.n_row = n_row
        self.pde_fn = pde_fn
        if xmin is None:
            self.xmin = [- L / 2, - H / 2]
        else:
            self.xmin = xmin
        nets = [dde.nn.FNN([input_dim] + num_layers * [hidden_dim] + [output_dim], "tanh", "Glorot normal") for _ in range(n_col * n_row)]
        self.nets = nn.ModuleList(nets)
        self.lower_bs = None
        self.eval_mode = False

    # check if index is legal
    def check_index(self, i, j):
        return (0 <= i < self.n_row) and \
            (0 <= j < self.n_col)

    # loss at interface 
    # x (this) and y (other) are 2 neighbors
    def loss_at_inter(self, x, y, pde_x, pde_y):
        loss_discont = torch.sum((x - (x + y) / 2) ** 2, dim=1, keepdim=True) 
        loss_pde_discont = torch.sum((pde_x - pde_y) ** 2, dim=1, keepdim=True)
        return torch.sqrt(loss_discont + loss_pde_discont)

    def append_loss(self, i, j, delta_i=0, delta_j=0):
        self.residuals_inter.append(self.loss_at_inter(
            self.subdomain_res[i][j],
            self.subdomain_res[i+delta_i][j+delta_j],
            self.subdomain_pde_res[i][j],
            self.subdomain_pde_res[i+delta_i][j+delta_j]
        ))
    
    def set_eval(self):
        '''
        Set the mode to evaluation, which does not calculate interface losses.
        '''
        self.eval_mode = True
    
    def set_training(self):
        '''
        Set the mode to training.
        '''
        self.eval_mode = False

    def forward(self, x) -> torch.Tensor:
        # init
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
            self.subdomain_res = [[None for _ in range(self.n_col)] for _ in range(self.n_row)]
            self.subdomain_pde_res = [[None for _ in range(self.n_col)] for _ in range(self.n_row)]
            self.residuals_inter = []
        # calculation results
        tot_res = 0.
        spatial_x = x[:, :2]
        for i in range(self.n_row):
            for j in range(self.n_col):
                indicator_res = torch.prod(
                    torch.logical_and(
                        self.lower_bs[i][j] <= spatial_x,
                        spatial_x <= self.upper_bs[i][j]
                    ), dim=1, keepdim=True
                )
                nn_res = self.nets[i * self.n_col + j](x)
                tot_res += indicator_res * nn_res
                self.subdomain_res[i][j] = nn_res
                if not self.eval_mode:
                    pde_res = self.pde_fn(x, nn_res)
                    if isinstance(pde_res, list):
                        pde_res = torch.cat(pde_res, dim=1)
                    self.subdomain_pde_res[i][j] = pde_res
        if self.eval_mode:
            return tot_res
        # residuals at interface
        self.residuals_inter.clear()
        for i in range(self.n_row):
            for j in range(self.n_col):
                if self.check_index(i, j-1):
                    self.append_loss(i, j, delta_j=-1)
                if self.check_index(i, j+1):
                    self.append_loss(i, j, delta_j=1)
                if self.check_index(i-1, j):
                    self.append_loss(i, j, delta_i=-1)
                if self.check_index(i+1, j):
                    self.append_loss(i, j, delta_i=1)
        return torch.cat((tot_res, torch.cat(self.residuals_inter, dim=1)), dim=1)
