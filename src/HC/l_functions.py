from typing import Callable
import numpy as np
import torch
from torch import optim, nn
import deepxde as dde
from src.utils.utils import cart2pol_pt, lineseg_dists

class LFunctionBase:
    """
    Lambda function.
    """
    def __init__(self, X, geom) -> None:
        self.X = X
        self.geom = geom
        self.alpha = None

    def get_dist(self, x):
        '''
        Calculate the extended distance to x (minibatch).
        '''
        del x

    def get_alpha(self):
        r'''
        Calculate the parameter $\alpha$.
        '''
        dists = self.get_dist(torch.tensor(self.X).float()).detach().cpu().numpy()
        dists_ = dists[~self.geom.on_boundary(self.X)]
        # make sure when it comes to the other nearest boundary
        # the coefficient goes down to exp(-5)
        self.alpha =  5 / np.min(dists_)
    
    def __call__(self, x):
        return torch.exp(-self.alpha * self.get_dist(x))


class LFunctionDisk(LFunctionBase):
    """
    Lambda function for a 2D disk.
    """
    def __init__(self, X, disk, inner=True) -> None:
        super().__init__(X, disk)
        self.center = torch.tensor(disk.center)
        self.radius = disk.radius
        self.inner = inner
        self.get_alpha()
    
    def get_dist(self, x):
        x = x[:, :2]
        if self.inner:
            return torch.linalg.norm(x - self.center, dim=1, keepdim=True) - self.radius
        else:
            return self.radius - torch.linalg.norm(x - self.center, dim=1, keepdim=True)


class LFunctionRectangle(LFunctionBase):
    """
    Lambda function for a 2D rectangle.
    """
    def __init__(self, X, rec, m_function: Callable) -> None:
        super().__init__(X, rec)
        self.xmin = rec.xmin
        self.xmax = rec.xmax
        self.m_function = m_function
        self.get_alpha()
    
    def get_dist(self, x):
        dist = torch.stack([
            x[:, 0] - self.xmin[0], - x[:, 0] + self.xmax[0],
            x[:, 1] - self.xmin[1], - x[:, 1] + self.xmax[1],
        ], dim=1)
        return self.m_function(dist)


class LFunctionOpenRectangle(LFunctionBase):
    """
    Lambda function for a 2D (right) open rectangle.\n
    |------------------\n
    |\n
    |------------------
    """
    def __init__(self, X, rec, m_function: Callable) -> None:
        super().__init__(X, rec)
        self.xmin = rec.xmin
        self.xmax = rec.xmax
        self.m_function = m_function
        self.get_alpha()
    
    def get_dist(self, x):
        dist = torch.stack([
            x[:, 0] - self.xmin[0],
            x[:, 1] - self.xmin[1], - x[:, 1] + self.xmax[1],
        ], dim=1)
        return self.m_function(dist)


class LFunctionAxisLine(LFunctionBase):
    """
    Lambda function for a line perpendicular to the axis.
    """
    def __init__(self, X, geom, x_0, j, is_left=True) -> None:
        '''
        x_0 - intersection point
        j - axis number (start from zero)
        is_left - left or right boundary
        '''
        super().__init__(X, geom)
        self.x_0 = x_0
        self.j = j
        self.is_left = is_left
        self.get_alpha()
    
    def get_dist(self, x):
        if self.is_left:
            return x[:, self.j:self.j+1] - self.x_0
        else:
            return -x[:, self.j:self.j+1] + self.x_0


class DistNet(nn.Module):
    """
    Network to produce a prediction of distance.
    """
    def __init__(self, reference_points) -> None:
        super(DistNet, self).__init__()
        self.reference_points = [
            torch.tensor(reference_point) 
            for reference_point in reference_points
        ]
        self.net = dde.nn.FNN([2 * len(reference_points)] + 3 * [30] + [1], 
            "tanh", "Glorot normal")

    def forward(self, x) -> torch.Tensor:
        x_polars = []
        for reference_point in self.reference_points:
            delta_x = x - reference_point
            x_polars.extend(cart2pol_pt(delta_x[:, 0:1], delta_x[:, 1:]))
        x_polars = torch.cat(x_polars, dim=1)
        dist_pred = self.net(x_polars)
        return dist_pred


class LFunctionPolygon(LFunctionBase):
    """
    Lambda function for a 2D polygon.
    """
    def __init__(self, X, polygon, spatial_domain) -> None:
        super().__init__(X, polygon)
        self.polygon = polygon
        self.vertices_left = polygon.vertices
        self.vertices_right = np.roll(polygon.vertices, 1, axis=0)
        center_1 = np.mean(polygon.vertices[polygon.vertices[:,0]<0.5,:], axis=0)
        center_2 = np.mean(polygon.vertices[polygon.vertices[:,0]>=0.5,:], axis=0)
        # sample points
        eps = 0.01
        self.bbox = dde.geometry.CSGDifference(
            dde.geometry.Rectangle(
                xmin=[np.min(polygon.vertices[:, 0]) - eps, np.min(polygon.vertices[:, 1]) - eps],
                xmax=[np.max(polygon.vertices[:, 0]) + eps, np.max(polygon.vertices[:, 1]) + eps]
        ), polygon)
        self.spatial_domain = spatial_domain
        X, dists = self.sample_points(1024 * 6)
        self.model = DistNet([center_1, center_2])
        X = torch.tensor(X).float()
        dists = torch.tensor(dists).float()
        self.train(X, dists)
        self.get_alpha()
    
    def sample_points(self, n):
        points = np.concatenate((
            self.bbox.random_points(n * 5 // 6),
            self.spatial_domain.random_points(n * 1 // 6)
        )) 
        dists = []
        for point in points:
            dists.append(
                [np.min(lineseg_dists(point, self.vertices_left, self.vertices_right))]
            )
        return points, np.array(dists)

    def loss_fn(self, dists, Y):
        loss = Y - dists
        return torch.mean(torch.abs(loss))

    def train(self, X, dists):
        print("Training extended dist for 2D polygon...")
        n_epochs = 10000
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=100, factor=0.75, min_lr=1e-5
        )
        for i in range(n_epochs):
            Y = self.model(X)
            loss = self.loss_fn(dists, Y)
            # Backpropagation
            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            if (i+1) % 1000 == 0 or i == 0:
                print(f"[Epoch {i+1}/{n_epochs}] loss: {loss.item():>7f}")
        # test
        X, dists = self.sample_points(1024)
        Y = self.get_dist(torch.tensor(X).float()).detach().cpu().numpy()
        print(f"Finish training!\nTesting loss: {np.mean(np.abs(Y - dists)):>7f}")

    def get_dist(self, x):
        x = x[:, :2]
        return self.model(x)
