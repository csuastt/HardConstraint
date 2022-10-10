import numpy as np
from deepxde.geometry.geometry import Geometry

class RecDiskDomain(Geometry):
    """
    Rectangular outer boundary & Circular inner boundary
    """

    def __init__(self, rec, disks):
        super().__init__(rec.dim, rec.bbox, rec.diam)
        self.rec = rec
        self.disks = disks

    def inside(self, x):
        inside_all = self.rec.inside(x)
        for disk in self.disks:
            inside_all = np.logical_and(inside_all, ~disk.inside(x))
        return inside_all

    def on_boundary(self, x):
        on_boundary_all = self.rec.on_boundary(x)
        for disk in self.disks:
            on_boundary_all = np.logical_or(on_boundary_all, disk.on_boundary(x))
        return on_boundary_all

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim))
        i = 0
        while i < n:
            tmp = self.rec.random_points(n, random=random)
            for disk in self.disks:
                tmp = tmp[~disk.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim))
        i = 0
        while i < n:

            tmp = self.rec.random_boundary_points(n, random=random)
            for disk in self.disks:
                disk_boundary_potins = disk.random_boundary_points(n, random=random)
                tmp = np.concatenate((tmp, disk_boundary_potins))
            
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x
    
    def boundary_normal(self, x):
        n = self.rec.on_boundary(x).reshape(-1, 1) * self.rec.boundary_normal(x)
        for disk in self.disks:
            n += disk.on_boundary(x).reshape(-1, 1) * -disk.boundary_normal(x)
        return n