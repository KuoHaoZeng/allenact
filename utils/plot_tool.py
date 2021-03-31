import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

plt.style.use("seaborn")
_color_set = ["blue", "green", "red", "magenta", "cyan", "yellow", "gray", "black", "white"]

def plot_line(ax, x, y, s1, s2, color="blue", label="", linewidth=1.0, alpha=0.5):
    assert alpha > 0.3
    ax.plot(x, y, alpha=alpha, color=color, label=label, linewidth=linewidth)
    ax.fill_between(x, s1, s2, color=color, alpha=alpha-0.3)
    return ax

def plot_mean_with_std(ax, x, y, color, label=""):
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    ax = plot_line(ax, x, mean, mean - std, mean + std, color, label)
    return ax

def plot_smooth_mean_with_std(ax, x, y, color, label="", ratio=0.1):
    window = int(float(len(y[0])) * ratio)
    if window % 2 == 0:
        window -= 1
    smooth_y = []
    for d in y:
        new_y = savgol_filter(d, window, 3)
        smooth_y.append(new_y)
    mean = np.mean(smooth_y, axis=0)
    std = np.std(smooth_y, axis=0)
    ax = plot_line(ax, x, mean, mean - std, mean + std, color, label)
    return ax

def plot_mean_with_upper_and_lower(ax, x, y, color, label=""):
    mean = np.mean(y, axis=0)
    upper = np.max(y, axis=0)
    lower = np.min(y, axis=0)
    ax = plot_line(ax, x, mean, lower, upper, color, label)
    return ax

def plot_smooth_mean_with_upper_and_lower(ax, x, y, color, label="", ratio=0.1):
    window = int(float(len(y[0])) * ratio)
    if window % 2 == 0:
        window -= 1
    smooth_y = []
    for d in y:
        new_y = savgol_filter(d, window, 3)
        smooth_y.append(new_y)
    mean = np.mean(smooth_y, axis=0)
    upper = np.max(smooth_y, axis=0)
    lower = np.min(smooth_y, axis=0)
    ax = plot_line(ax, x, mean, lower, upper, color, label)
    return ax

def plot_many_series(data, key, num_samples=3, plot_func=plot_mean_with_std, labels=["", ""]):
    assert len(data) // num_samples == len(labels)
    fig, ax = plt.subplots(figsize=(16, 8))
    x, y = [], []
    color_ind = 0
    for k, v in data.items():
        x_ = [ele[0] for ele in v[key]]
        y_ = [ele[1] for ele in v[key]]
        x.append(x_)
        y.append(y_)
        if len(x) == num_samples:
            x, y = sync_data(x, y)
            ax = plot_func(ax, x[0], y, _color_set[color_ind], labels[color_ind])
            x, y = [], []
            color_ind += 1
    return ax, fig

def sync_data(x, y):
    m = len(x[0])
    for ele in x:
        m = min(m, len(ele))
    new_x, new_y = [], []
    for i in range(len(x)):
        new_x.append(x[i][-m:])
        new_y.append(y[i][-m:])
        if i > 0:
            assert len(new_x[-1]) == len(new_x[-2])
    return new_x, new_y