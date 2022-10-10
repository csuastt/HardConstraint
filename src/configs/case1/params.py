from src.utils.utils import load_params_disks

# path
data_path = "data/case1_pack.txt"
model_path_prefix = "model/case1_"

# parameters for PDEs
H = 24
L = 16
k = 1. # thermal conductivity

# parameters of the geometry
# read from the input file
disk_centers, disk_rs = load_params_disks(data_path)
disk_centers_c = disk_centers[:-6]
disk_rs_c = disk_rs[:-6]
disk_centers_w = disk_centers[-6:]
disk_rs_w = disk_rs[-6:]
