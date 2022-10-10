import torch
from src.PFNN.pfnn import PFNN
from src.configs.case2.hc import l_gamma_openrec, l_gamma_right

class PFNN_NS(PFNN):
    """
    Penalty-Free Neural Network for NS equation.
    """

    def forward(self, x) -> torch.Tensor:
        raw_output = self.net(x)
        u_hat = raw_output[:, 0] * l_gamma_openrec.get_dist(x) + 1.
        v_hat = raw_output[:, 1] * l_gamma_openrec.get_dist(x) + 0.
        p_hat = raw_output[:, 2] * l_gamma_right.get_dist(x) + 1.
        return torch.cat([u_hat, v_hat, p_hat], dim=1)