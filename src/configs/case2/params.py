from src.utils.utils import load_airfoil_points

# path
data_path = "data/case2_airfoil.txt"
airfoil_path = "data/w1015.dat"
model_path_prefix = "model/case2_"

# parameters for PDEs
xmin = [-2, -2]
xmax = [6, 2]
Re = 50.
nu = 1 / Re # viscosity

# the anchor points of the airfoil
anchor_points = load_airfoil_points(airfoil_path)
