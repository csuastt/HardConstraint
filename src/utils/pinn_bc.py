import deepxde.utils as utils
import deepxde.backend as bkd
from deepxde.icbc.boundary_conditions import BC, npfunc_range_autocache

class NormalBC(BC):
    """Normal boundary conditions: y_{component} * n_1(x) + y_{component+1} * n_2(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end):
        values = self.func(X, beg, end)
        n = self.boundary_normal(X, beg, end)
        y = outputs[beg:end, self.component : self.component + 2]
        return bkd.sum(y * n, 1, keepdims=True) - values