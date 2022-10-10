import deepxde as dde
from src.configs.case1.params import *
from src.utils.pinn_geometry import RecDiskDomain

rec = dde.geometry.Rectangle(xmin=[-L/2, -H/2], xmax=[L/2, H/2])
disks_c = [dde.geometry.Disk(center=disk_centers_c[i], radius=disk_rs_c[i]) for i in range(len(disk_centers_c))]
disks_w = [dde.geometry.Disk(center=disk_centers_w[i], radius=disk_rs_w[i]) for i in range(len(disk_centers_w))]
spatial_domain = RecDiskDomain(rec, disks_c + disks_w)
time_domain = dde.geometry.TimeDomain(0, 1)
spatial_time_domain = dde.geometry.GeometryXTime(spatial_domain, time_domain)


def pde_pinn(x, T):
    T_t = dde.grad.jacobian(T, x, i=0, j=2)
    T_xx = dde.grad.hessian(T, x, i=0, j=0)
    T_yy = dde.grad.hessian(T, x, i=1, j=1)
    return T_t - k * (T_xx + T_yy)


rec_bc = dde.icbc.RobinBC(
    spatial_time_domain, lambda _, T: 1e-1 - T, 
    lambda x, on_bc: on_bc and rec.on_boundary(x[:2])
)

bc_disks = []
for i in range(len(disks_c)):
    bc_disks.append(
        dde.icbc.RobinBC(
            spatial_time_domain, lambda _, T: 5. - T, 
            lambda x, on_bc, j=i: on_bc and disks_c[j].on_boundary(x[:2])
        )
    )

for i in range(len(disks_w)):
    bc_disks.append(
        dde.icbc.RobinBC(
            spatial_time_domain, lambda _, T: 1. - T, 
            lambda x, on_bc, j=i: on_bc and disks_w[j].on_boundary(x[0:2])
        )
    )

ic = dde.icbc.IC(
    spatial_time_domain, lambda _: 1e-1, 
    lambda _, on_initial: on_initial
)

ic_bcs = [rec_bc] + bc_disks + [ic]

# parameters for PINN
# moving average in learning rate annealing
lr_alpha = 0.1
# num pdes
num_pdes = 1
num_bcs = len(ic_bcs)