import numpy as np
import deepxde as dde


class Subdomains:
    """
    Generate subdomains and their interface conditions for the xPINN.
    ## Parameter
    L - length\n
    H - height\n
    n_col - the number of columns of the sub-domains\n
    n_row - the number of rows of the sub-domains\n
    xmin - the offset of the left-bottom point\n
    domain - problem domain\n
    num_output - the number outputs
    """
    def __init__(
        self, L, H, n_col, n_row, num_output, spatial_temporal_domain, xmin=None, temporal_domain=None
    ) -> None:
        self.L = L
        self.H = H
        self.n_col = n_col
        self.n_row = n_row
        self.num_output = num_output
        self.domain = spatial_temporal_domain
        if xmin is None:
            xmin = [- L / 2, - H / 2]
        self.lower_bs = [
            [
                [L * j / n_col + xmin[0],
                    H * i / n_row + xmin[1]]
                for j in range(n_col)
            ] for i in range(n_row)
        ]
        self.upper_bs = [
            [
                [L * (j+1) / n_col + xmin[0],
                    H * (i+1) / n_row  + xmin[1]]
                for j in range(n_col)
            ] for i in range(n_row)
        ]
        self.sub_blocks = [
            [
                dde.geometry.Rectangle(
                    xmin=self.lower_bs[i][j], xmax=self.upper_bs[i][j]
                ) if temporal_domain is None else
                dde.geometry.GeometryXTime(
                    dde.geometry.Rectangle(
                        xmin=self.lower_bs[i][j], xmax=self.upper_bs[i][j]
                    ), temporal_domain
                )
                for j in range(n_col)
            ] for i in range(n_row)
        ]

    def generate_interface_points(self, n):
        points = []
        for i in range(self.n_row):
            for j in range(self.n_col):
                points.append(
                    self.sub_blocks[i][j].random_boundary_points(n)
                )
        points = np.concatenate(points, axis=0)
        np.random.shuffle(points)
        return points[:n, :]

    # check if index is legal
    def check_index(self, i, j):
        return (0 <= i < self.n_row) and \
            (0 <= j < self.n_col)

    def generate_interface_conditions(self):
        interface_conditions = []
        cnt = self.num_output
        for i in range(self.n_row):
            for j in range(self.n_col):
                if self.check_index(i, j-1):
                    interface_conditions.append(
                        dde.icbc.DirichletBC(
                            self.domain, lambda _: 0., 
                            lambda x, on_bc: np.isclose(x[0], self.lower_bs[i][j][0]) and \
                                (self.lower_bs[i][j][1] <= x[1] <= self.upper_bs[i][j][1]), 
                            component=cnt
                        )
                    )
                    cnt += 1
                if self.check_index(i, j+1):
                    interface_conditions.append(
                        dde.icbc.DirichletBC(
                            self.domain, lambda _: 0., 
                            lambda x, on_bc: np.isclose(x[0], self.upper_bs[i][j][0]) and \
                                (self.lower_bs[i][j][1] <= x[1] <= self.upper_bs[i][j][1]), 
                            component=cnt
                        )
                    )
                    cnt += 1
                if self.check_index(i-1, j):
                    interface_conditions.append(
                        dde.icbc.DirichletBC(
                            self.domain, lambda _: 0., 
                            lambda x, on_bc: np.isclose(x[1], self.lower_bs[i][j][1]) and \
                                (self.lower_bs[i][j][0] <= x[0] <= self.upper_bs[i][j][0]),  
                            component=cnt
                        )
                    )
                    cnt += 1
                if self.check_index(i+1, j):
                    interface_conditions.append(
                        dde.icbc.DirichletBC(
                            self.domain, lambda _: 0., 
                            lambda x, on_bc: np.isclose(x[1], self.upper_bs[i][j][1]) and \
                                (self.lower_bs[i][j][0] <= x[0] <= self.upper_bs[i][j][0]), 
                            component=cnt
                        )
                    )
                    cnt += 1
        return interface_conditions
