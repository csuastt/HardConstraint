import torch
import numpy as np
from src.utils.torch_interp import Interp1d
from src.utils.utils import cart2pol_np, cart2pol_pt


class NormalFunctionDisk:
    """
    Normal function for a 2D disk.
    """
    def __init__(self, center, inner=True) -> None:
        self.center = torch.tensor(center)
        self.inner = inner
    
    def __call__(self, x):
        '''
        Calcute the extended (outer) normal at x (minibatch)
        '''
        x = x[:, :2]
        if self.inner:
            d_x = self.center - x
        else:
            d_x = x - self.center
        return d_x / torch.linalg.norm(d_x, dim=1, keepdim=True)


class NormalFunctionSphere:
    """
    Normal function for a ND sphere.
    """
    def __init__(self, center, inner=True) -> None:
        self.n_dim = len(center)
        self.center = torch.tensor(center)
        self.inner = inner
    
    def __call__(self, x):
        '''
        Calcute the extended (outer) normal at x (minibatch)
        '''
        x = x[:, :self.n_dim]
        if self.inner:
            d_x = self.center - x
        else:
            d_x = x - self.center
        return d_x / torch.linalg.norm(d_x, dim=1, keepdim=True)


class NormalFunctionRectangle:
    """
    Normal function for a 2D rectangle (centered at (0,0), outer boundary).
    """
    def __init__(self, H, L) -> None:
        self.H = H
        self.L = L

    def __call__(self, x):
        '''
        Calcute the extended (outer) normal at x (minibatch)
        '''
        x = x[:, :2]
        k = self.H / self.L
        n = torch.zeros_like(x)
        n[torch.where(torch.isclose(x[:,0], torch.tensor(0.)))] = torch.tensor([0., 1.])
        n[torch.where(torch.logical_and(torch.logical_or(x[:,1] >= x[:,0] * k, x[:,1] <= -x[:,0] * k), x[:,1] >= 0))] = torch.tensor([0., 1.])
        n[torch.where(torch.logical_and(torch.logical_or(x[:,1] >= x[:,0] * k, x[:,1] <= -x[:,0] * k), x[:,1] <= 0))] = torch.tensor([0., -1.])
        n[torch.where(torch.logical_and(torch.logical_and(x[:,1] <= x[:,0] * k, x[:,1] >= -x[:,0] * k), x[:,0] >= 0))] = torch.tensor([1., 0.])
        n[torch.where(torch.logical_and(torch.logical_and(x[:,1] <= x[:,0] * k, x[:,1] >= -x[:,0] * k), x[:,0] <= 0))] = torch.tensor([-1., 0.])
        return n

class NormalFunctionPolygon:
    """
    Normal function for a 2D polygon.
    """
    def __init__(self, polygon, inner=True) -> None:
        self.polygon = polygon
        self.inner = inner
        # reference center
        self.center_np = np.mean(
            polygon.vertices, axis=0
        )
        self.center = torch.tensor(self.center_np)
        # middle points in the polygon
        X = (polygon.vertices + np.roll(polygon.vertices, 1, axis=0)) / 2
        if self.inner:
            normals = -self.polygon.boundary_normal(X)
        else:
            normals = self.polygon.boundary_normal(X)
        thetas = self.get_thetas_np(X)
        normal_thetas = cart2pol_np(normals[:, 0], normals[:, 1])[1]
        # correct the thetas
        eps = 1e-66
        normal_thetas[0] = -np.pi
        normal_thetas = np.concatenate((normal_thetas, [np.pi]))
        thetas[0, :] = [0]
        thetas = np.concatenate((thetas, [[-eps]]))
        sorted_indices = np.argsort(thetas[:, 0])
        thetas = thetas[sorted_indices, :]
        normal_thetas = normal_thetas[sorted_indices]
        thetas[0, :] = [-np.pi]
        thetas = np.concatenate((thetas, [[np.pi]]))
        normal_thetas = np.concatenate((normal_thetas, [0]))
        # interpolation
        self.thetas = torch.tensor(thetas.T)
        self.normal_thetas = torch.tensor(normal_thetas)
        self.interp = Interp1d()

    def get_thetas_pt(self, x):
        delta_x = x - self.center
        theta_x = cart2pol_pt(delta_x[:, 0:1], delta_x[:, 1:])[1]
        return theta_x
    
    def get_thetas_np(self, x):
        delta_x = x - self.center_np
        theta_x = cart2pol_np(delta_x[:, 0:1], delta_x[:, 1:])[1]
        return theta_x

    def __call__(self, x):
        '''
        Calcute the extended (outer) normal at x (minibatch)
        '''
        x = x[:, :2]
        x_thetas = self.get_thetas_pt(x)
        normal_thetas = self.interp(
            self.thetas, self.normal_thetas, 
            torch.transpose(x_thetas, 0, 1)
        )
        normal_thetas = torch.transpose(normal_thetas, 0, 1)
        return torch.concat([torch.cos(normal_thetas), torch.sin(normal_thetas)], dim=1)
