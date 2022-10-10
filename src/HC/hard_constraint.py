import torch
from torch import nn
from src.utils.nn_wrapper import NNWrapper


class HardConstraintRobin2D(nn.Module):
    """
    Build hard constraint formula for Robin BCs in 2D.
    """
    def __init__(self, f_normal, NN: NNWrapper, f_a, f_b, f_g) -> None:
        '''
        f_normal - function for calculating normals.\n
        NN - a neural network.\n
        ... - parameters for the Robin BCs
        '''
        super(HardConstraintRobin2D, self).__init__()
        self.f_normal = f_normal
        self.NN = NN
        self.f_a = f_a
        self.f_b = f_b
        self.f_g = f_g
        self.normal_res = None
    
    def get_u(self, x):
        if self.normal_res is None:
            self.normal_res = self.f_normal(x)
        ns = self.normal_res
        hs = self.NN(x)
        a = self.f_a(x)
        b = self.f_b(x)
        res = b * (-ns[:, 1:2] * hs[:, 0:1] + ns[:, 0:1] * hs[:, 1:2]) + a * self.f_g(x)
        return res / (a ** 2 + b ** 2)

    def get_p_1(self, x):
        if self.normal_res is None:
            self.normal_res = self.f_normal(x)
        ns = self.normal_res
        hs = self.NN(x)
        a = self.f_a(x)
        b = self.f_b(x)
        res = b * (ns[:, 1:2] * hs[:, 2:3] + ns[:, 0:1] * self.f_g(x)) - a * hs[:, 1:2]
        return res / (a ** 2 + b ** 2)

    def get_p_2(self, x):
        if self.normal_res is None:
            self.normal_res = self.f_normal(x)
        ns = self.normal_res
        hs = self.NN(x)
        a = self.f_a(x)
        b = self.f_b(x)
        res = b * (-ns[:, 0:1] * hs[:, 2:3] + ns[:, 1:2] * self.f_g(x)) + a * hs[:, 0:1]
        return res / (a ** 2 + b ** 2)

    def clear_res(self):
        self.NN.clear_res()
        self.normal_res = None
    
    def save(self, path_prefix: str, name: str):
        torch.save(self.NN, path_prefix + name)

    def load(self, path_prefix: str, name: str):
        self.NN = torch.load(path_prefix + name)


class HardConstraintNeumann2D(nn.Module):
    """
    Build hard constraint formula for Neumann BCs in 2D.
    """
    def __init__(self, f_normal, NN: NNWrapper, f_g) -> None:
        '''
        f_normal - function for calculating normals.\n
        NN - a neural network.\n
        ... - parameters for the Neumann BCs
        '''
        super(HardConstraintNeumann2D, self).__init__()
        self.f_normal = f_normal
        self.NN = NN
        self.f_g = f_g
        self.normal_res = None

    def get_p_1(self, x):
        if self.normal_res is None:
            self.normal_res = self.f_normal(x)
        ns = self.normal_res
        h = self.NN(x)
        return ns[:, 1:2] * h + ns[:, 0:1] * self.f_g(x)

    def get_p_2(self, x):
        if self.normal_res is None:
            self.normal_res = self.f_normal(x)
        ns = self.normal_res
        h = self.NN(x)
        return -ns[:, 0:1] * h + ns[:, 1:2] * self.f_g(x)

    def clear_res(self):
        self.NN.clear_res()
        self.normal_res = None

    def save(self, path_prefix: str, name: str):
        torch.save(self.NN, path_prefix + name)

    def load(self, path_prefix: str, name: str):
        self.NN = torch.load(path_prefix + name)


class HardConstraintNeumannND(nn.Module):
    """
    Build hard constraint formula for Neumann BCs in ND.
    """
    def __init__(self, f_normal, NN: NNWrapper, f_g) -> None:
        '''
        f_normal - function for calculating normals.\n
        NN - a neural network.\n
        ... - parameters for the Neumann BCs
        '''
        super(HardConstraintNeumannND, self).__init__()
        self.f_normal = f_normal
        self.NN = NN
        self.f_g = f_g
        self.res = None

    def get_p_i(self, x, i):
        if self.res is None:
            ns = self.f_normal(x)
            hs = self.NN(x)
            self.res = hs + ns * (self.f_g(x) - torch.sum(ns * hs, dim=1, keepdim=True))
        return self.res[:, i:i+1]

    def clear_res(self):
        self.NN.clear_res()
        self.res = None

    def save(self, path_prefix: str, name: str):
        torch.save(self.NN, path_prefix + name)

    def load(self, path_prefix: str, name: str):
        self.NN = torch.load(path_prefix + name)