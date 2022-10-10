import torch
from torch import nn

class HardConstraintCollector(nn.Module):
    """
    Parametrize a single state variable u_i.
    """
    def __init__(self, i, M, ls, us, N) -> None:
        '''
        i - index of the state variable.\n
        M - M function, taking B x m tensors as inputs, where B is the batch size, 
        and m is the number of boundaries, outputing B x 1 tensors.\n
        ls - a list of m callable objects (lambda functions), each taking B x d tensors as inputs,
            where d is the dimensionality, outputing B x 1 tensors.\n
        us - a list of m callable objects (the general solutions of u_i at each boundary, B x 1), 
            taking B x d tensors as inputs.\n
        N - a callable object to generate the raw output (B x d'), 
            taking B x d tensors and the index as inputs.
        '''
        super(HardConstraintCollector, self).__init__()
        self.i = i
        self.M = M
        self.ls = ls
        self.us = us
        self.N = N
    
    def forward(self, x) -> torch.Tensor:
        """
        Map the coordinates to the state variable u_i.\n
        x - coordinates, a B x d tensor, where B is the batch size, and d is the dimensionality.\n
        Return a B x 1 tensor corresponding to u_i.
        """
        dists = torch.cat([l.get_dist(x) for l in self.ls], dim=1) # output: B x m
        u_res = self.M(dists) * self.N(x, self.i) # output: B x 1
        for j in range(len(self.ls)):
            u_res += self.ls[j](x) * self.us[j](x)
        return u_res
