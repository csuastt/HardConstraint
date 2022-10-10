import matplotlib.pyplot as plt
from array import array
from typing import Callable
import torch
from torch import nn
import numpy as np
from scipy import interpolate


epsilon = 1e-6


def load_params_disks(data_path: str) -> tuple[list[list[float]], list[float]]:
    '''
    Load the parameters of the disks from the first line of the input file.
    '''
    disk_centers = []
    disk_rs = []
    with open(data_path, "r") as f:
        line = f.readline()
        ls = line.split()
        for i in range(1, len(ls), 3):
            disk_centers.append([float(ls[i]), float(ls[i+1])])
            disk_rs.append(float(ls[i+2]))
    return disk_centers, disk_rs


def load_airfoil_points(airfoil_path: str) -> list[list[float]]:
    points = []
    with open(airfoil_path, "r") as f:
        for line in f.readlines():
            line = line.split()
            if len(line) == 2:
                points.append(
                    [float(line[0]),
                    float(line[1])]
                )
    return np.array(points)


def load(data_path: str, dim_X: int) -> tuple[array, array]:
    X = []
    Y = []
    with open(data_path, "r") as f:
        for line in f.readlines():
            ls = line.split()
            if ls[0] == '%':
                continue
            ls = [float(s) for s in ls]
            X.append(ls[:dim_X])
            Y.append(ls[dim_X:])
    return np.array(X), np.array(Y)


def load_time(data_path: str, dim_X: int) -> tuple[array, array]:
    '''
    Load time-dependent data
    '''
    X = []
    Y = []
    with open(data_path, "r") as f:
        for line in f.readlines():
            ls = line.split()
            if ls[0] == '%':
                continue
            ls = [float(s) for s in ls]
            X.extend([ls[:dim_X] + [i/10] for i in range(11)])
            Y.extend([[ls[dim_X + i]] for i in range(11)])
    return np.array(X), np.array(Y)


def test(data_path: str, dim_X: int, model: nn.Module) -> None:
    test_X, test_Y = load(data_path, dim_X)
    pred_Y = model(torch.tensor(test_X).float()).detach().cpu().numpy()
    for j in range(test_Y.shape[1]):
        abs_e = np.absolute(pred_Y[:, j] - test_Y[:, j])
        m_abs_e = np.mean(abs_e)
        print("Mean abs error of u_%d: %.4f"%(j, m_abs_e))
        wmape = np.sum(abs_e) / np.sum(np.absolute(test_Y[:, j]))
        print("Weighted mean abs percentage error of u_%d: %.4f"%(j, wmape))
        # plot the heat map
        vmin = max(np.min(
            test_Y[:, j]
        ), 0)
        vmax = np.max(
            test_Y[:, j]
        )
        plot_heatmap(
            test_X[:,0].reshape(-1), test_X[:,1].reshape(-1),
            test_Y[:, j], "outs/heatmap_exact_u_%d.png"%j,
            title="Heatmap of exact u_%d"%j, vmin=vmin, vmax=vmax
        )
        plot_heatmap(
            test_X[:,0].reshape(-1), test_X[:,1].reshape(-1),
            pred_Y[:, j], "outs/heatmap_pred_u_%d.png"%j,
            title="Heatmap of pred u_%d"%j, vmin=vmin, vmax=vmax
        )
        vmin = max(np.min(
            abs_e
        ), 0)
        plot_heatmap(
            test_X[:,0].reshape(-1), test_X[:,1].reshape(-1),
            abs_e, "outs/heatmap_r_abs_e_u_%d.png"%j,
            title="Heatmap of absolute error of u_%d"%j, vmin=vmin
        )
    # save the results
    with open("outs/result.txt", "w") as f:
        for i in range(test_Y.shape[0]):
            for j in range(test_X.shape[1]):
                f.write("%f "%test_X[i, j])
            for j in range(test_Y.shape[1]):
                f.write("%f "%pred_Y[i, j])
            f.write("\n")


def test_time(data_path: str, dim_X: int, model: nn.Module) -> None:
    test_X, test_Y = load_time(data_path, dim_X)
    pred_Y = model(torch.tensor(test_X).float()).detach().cpu().numpy()
    for j in range(test_Y.shape[1]):
        m_abs_es = []
        m_r_abs_es = []
        for t in range(11):
            t = t / 10
            index = np.where(np.isclose(test_X[:,2], t))
            pred_Y_t = pred_Y[index]
            test_Y_t = test_Y[index]
            test_X_t = test_X[index][:, :2]
            abs_e = np.absolute(pred_Y_t[:, j] - test_Y_t[:, j])
            m_abs_e = np.mean(abs_e)
            print("Mean abs error of u_%d (t=%.1f): %.4f"%(j, t, m_abs_e))
            m_abs_es.append(m_abs_e)
            r_abs_e = abs_e / np.absolute(test_Y_t[:, j])
            m_r_abs_e = np.mean(r_abs_e)
            print("Mean abs percentage error of u_%d (t=%.1f): %.4f"%(j, t, m_r_abs_e))
            m_r_abs_es.append(m_r_abs_e)
            # plot the heat map
            vmin = max(np.min(
                test_Y_t[:, j]
            ), 0)
            vmax = np.max(
                test_Y_t[:, j]
            )
            plot_heatmap(
                test_X_t[:,0].reshape(-1), test_X_t[:,1].reshape(-1),
                test_Y_t[:, j], "outs/heatmap_exact_u_%d (t=%.1f).png"%(j, t),
                title="Heatmap of exact u_%d (t=%.1f)"%(j, t), vmin=vmin, vmax=vmax
            )
            plot_heatmap(
                test_X_t[:,0].reshape(-1), test_X_t[:,1].reshape(-1),
                pred_Y_t[:, j], "outs/heatmap_pred_u_%d (t=%.1f).png"%(j, t),
                title="Heatmap of pred u_%d (t=%.1f)"%(j, t), vmin=vmin, vmax=vmax
            )
            vmin = max(np.min(
                r_abs_e
            ), 0)
            plot_heatmap(
                test_X_t[:,0].reshape(-1), test_X_t[:,1].reshape(-1),
                r_abs_e, "outs/heatmap_r_abs_e_u_%d (t=%.1f).png"%(j, t),
                title="Heatmap of absolute percentage error of u_%d (t=%.1f)"%(j, t), vmin=vmin
            )
        print("Overall mean abs error of u_%d: %.4f"%(j, np.mean(m_abs_es)))
        print("Overall mean abs percentage error of u_%d: %.4f"%(j, np.mean(m_r_abs_es)))
    # save the results
    with open("outs/result.txt", "w") as f:
        for i in range(test_X.shape[0] // 11):
            f.write("%f %f "%(test_X[i * 11, 0], test_X[i * 11, 1]))
            for j in range(11):
                for k in range(test_Y.shape[1]):
                    f.write("%f "%pred_Y[i * 11 + j, k])
            f.write("\n")


def test_time_with_reference_solution(
    reference_solution: Callable, test_X, 
    model: nn.Module
):
    test_Y = reference_solution(test_X)
    pred_Y = model(torch.tensor(test_X).float()).detach().cpu().numpy()
    for j in range(test_Y.shape[1]):
        m_abs_es = []
        m_r_abs_es = []
        for t in range(11):
            t = t / 10
            index = np.where(np.isclose(test_X[:,-1], t))
            pred_Y_t = pred_Y[index]
            test_Y_t = test_Y[index]
            abs_e = np.absolute(pred_Y_t[:, j] - test_Y_t[:, j])
            m_abs_e = np.mean(abs_e)
            print("Mean abs error of u_%d (t=%.1f): %.4f"%(j, t, m_abs_e))
            m_abs_es.append(m_abs_e)
            r_abs_e = abs_e / np.absolute(test_Y_t[:, j])
            m_r_abs_e = np.mean(r_abs_e)
            print("Mean abs percentage error of u_%d (t=%.1f): %.4f"%(j, t, m_r_abs_e))
            m_r_abs_es.append(m_r_abs_e)
        print("Overall mean abs error of u_%d: %.4f"%(j, np.mean(m_abs_es)))
        print("Overall mean abs percentage error of u_%d: %.4f"%(j, np.mean(m_r_abs_es)))


class Tester:
    test_X_while_train = None
    test_Y_while_train = None
    def test_while_train(data_path: str, dim_X: int, model: nn.Module) -> list:
        if Tester.test_X_while_train is None:
            Tester.test_X_while_train, Tester.test_Y_while_train = load(data_path, dim_X)
            Tester.test_X_while_train = torch.tensor(Tester.test_X_while_train).float()
        pred_Y = model(Tester.test_X_while_train).detach().cpu().numpy()
        m_abs_e_res, m_r_abs_e_res = [], []
        for j in range(Tester.test_Y_while_train.shape[1]):
            abs_e = np.absolute(pred_Y[:, j] - Tester.test_Y_while_train[:, j])
            m_abs_e = np.mean(abs_e)
            m_abs_e_res.append(m_abs_e)
            r_abs_e = abs_e / np.absolute(Tester.test_Y_while_train[:, j])
            m_r_abs_e = np.mean(r_abs_e)
            m_r_abs_e_res.append(m_r_abs_e)
        return m_abs_e_res, m_r_abs_e_res


def plot_distribution(data, xlabel, ylabel, path, title="", log_scal=True):
    '''
    plot the distribution of data with log-scale.
    '''
    plt.cla()
    plt.figure()
    _, bins, _ = plt.hist(data, bins=30)
    plt.close()
    plt.cla()
    plt.figure()
    logbins = np.logspace(np.log10(bins[0] + epsilon),np.log10(bins[-1] + epsilon),len(bins))
    plt.hist(data, bins=logbins)
    if log_scal:
        plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axvline(x=np.mean(data), c="r", ls="--", lw=2)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def plot_lines(
    data, xlabel, ylabel, labels,
    path, is_log=False, title="",
    sort_=False
):
    '''
    Lines
    '''
    plt.cla()
    plt.figure()
    for i in range(1, len(data)):
        if sort_:
            x = np.array(data[0])
            y = np.array(data[i])
            sorted_indices = np.argsort(x)
            sorted_x = x[sorted_indices]
            sorted_y = y[sorted_indices]
            plt.plot(sorted_x, sorted_y, label=labels[i-1])
        else:
            plt.plot(data[0], data[i], label=labels[i-1])
    plt.legend()
    if is_log:
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def plot_heatmap(
    x, y, z, path, vmin=None, vmax=None,
    title="", xlabel="x", ylabel="y"
):
    '''
    Plot heat map for a 3-dimension data
    '''
    plt.cla()
    plt.figure()
    xx = np.linspace(np.min(x), np.max(x))
    yy = np.linspace(np.min(y), np.max(y))
    xx, yy = np.meshgrid(xx, yy)
    yy = yy[::-1,:]

    vals = interpolate.griddata(np.array([x, y]).T, np.array(z), 
        (xx, yy), method='cubic')
    vals_0 = interpolate.griddata(np.array([x, y]).T, np.array(z), 
        (xx, yy), method='nearest')
    vals[np.isnan(vals)] = vals_0[np.isnan(vals)]

    if vmin is not None and vmax is not None:
        fig = plt.imshow(vals,
                extent=[np.min(x), np.max(x),np.min(y), np.max(y)],
                aspect="equal", interpolation="bicubic",
                vmin=vmin, vmax=vmax)
    elif vmin is not None:
        fig = plt.imshow(vals,
                extent=[np.min(x), np.max(x),np.min(y), np.max(y)],
                aspect="equal", interpolation="bicubic",
                vmin=vmin)
    else:
        fig = plt.imshow(vals,
                extent=[np.min(x), np.max(x),np.min(y), np.max(y)],
                aspect="equal", interpolation="bicubic")
    fig.axes.set_autoscale_on(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.savefig(path)
    plt.close()

def cart2pol_np(x, y):
    '''
    From Cartesian coordinates to polar coordinates (implemented by numpy).
    Return: (r, theta)
    '''
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r, theta)

def cart2pol_pt(x, y):
    '''
    From Cartesian coordinates to polar coordinates (implemented by pytorch).
    Return: (r, theta)
    '''
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return (r, theta)

def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from: anonymous

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                            .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)