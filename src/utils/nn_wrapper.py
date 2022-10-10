import torch
from torch import nn

class NNWrapper(nn.Module):
    """
    Wrapper a NN to avoid repeating calculations.
    """
    def __init__(self, NN: nn.Module) -> None:
        super(NNWrapper, self).__init__()
        self.NN = NN
        self.res = None # calculation result
    
    def clear_res(self) -> None:
        '''
        Clear the calculation result of the last batch.
        Note: call it at the beginning of a batch.
        '''
        self.res = None

    def forward(self, x, i=None) -> torch.Tensor:
        '''
        x - coordinate
        i - the ith column of the result will be returned (keepdim)
        '''
        if self.res is None:
            self.res = self.NN(x)
        if self.res.dim() == 1:
            assert i == 0 or i is None
            return self.res[:]
        if i is None:
            return self.res
        else:
            return self.res[:, i:i+1]
