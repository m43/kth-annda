import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import StrMethodFormatter
from pathlib import Path

horse = """               .,,.
             ,;;*;;;;,
            .-'``;-');;.
           /'  .-.  /*;;
         .'    \\d    \\;;               .;;;,
        / o      `    \\;    ,__.     ,;*;;;*;,
        \\__, _.__,'   \\_.-') __)--.;;;;;*;;;;,
         `""`;;;\\       /-')_) __)  `\' ';;;;;;
            ;*;;;        -') `)_)  |\\ |  ;;;;*;
            ;;;;|        `---`    O | | ;;*;;;
            *;*;\\|                 O  / ;;;;;*
           ;;;;;/|    .-------\\      / ;*;;;;;
          ;;;*;/ \\    |        '.   (`. ;;;*;;;
          ;;;;;'. ;   |          )   \\ | ;;;;;;
          ,;*;;;;\\/   |.        /   /` | ';;;*;
           ;;;;;;/    |/       /   /__/   ';;;
           '*jgs/     |       /    |      ;*;
                `""""`        `""""`     ;'"""  # Why jgs?


def extend_inputs_with_bias(inputs, value=1, features_axis=1) -> np.array:
    """
    Function returns given input extended by `value` at `axis`.

    :param inputs: inputs numpy array to extend; the input array will not be modified
    :param value: value to extend by
    :param features_axis: at which axis to extend. the value can either be 0 or 1
    :return: resulting extended array
    """
    ones = np.ones((inputs.shape[1 - features_axis], 1), inputs.dtype)
    if features_axis == 0:
        ones = ones.T

    return np.concatenate((inputs, value * ones), axis=features_axis)


def sse(outputs, targets):
    return np.sum((targets - outputs) ** 2) / 2


def mse(outputs, targets):
    return np.sum((targets - outputs) ** 2) / targets.size / 2


def mae(outputs, targets):
    return np.sum(np.abs((targets - outputs))) / targets.size


def standardize(x, mean, std):
    return (x - mean) / std


def standardize_dataset(x, mean, std):
    return (standardize(x[0], mean, std), x[1])


def destandardize(x, mean, std):
    return x * std + mean


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def sigmoid(x, beta):
    return 1. / (1. + np.exp(-beta * x))


def number_to_one_hoot(targets, digits=10):
    result = np.zeros((targets.size, digits))
    result[np.arange(targets.size), targets] = 1
    return result


def shuffle_two_arrays(a, b):
    shuffler = np.random.permutation(a.shape[-1])
    a = a[:, shuffler]
    b = b[:, shuffler]
    return a, b


def plot_metric(data, name, save, *more_data, point=None, show_plot=True):
    # plt.figure(figsize=(9, 6))
    x = np.arange(len(data))
    plt.title(name)
    plt.plot(x, data, color="green")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.6f}'))
    for d in more_data:
        plt.plot(x, d)
    if point is not None:
        plt.scatter(point[0], point[1], color="red")
    if save:
        plt.savefig(name, dpi=300)
    if show_plot:
        plt.show()


def accuracy(outputs, targets, targets_are_rows: bool):
    n = targets.shape[0] if targets_are_rows else targets.shape[1]
    return np.sum(np.all(np.equal(targets, outputs), axis=(1 if targets_are_rows else 0))) * 100 / n


class TwoClassDatasetGenerator:
    def __init__(self, m_a: tuple, n_a: int, sigma_a: tuple, m_b: tuple, n_b: int, sigma_b: tuple):
        self.m_a = m_a
        self.n_a = n_a
        self.sigma_a = sigma_a
        self.m_b = m_b
        self.n_b = n_b
        self.sigma_b = sigma_b

    def random_1(self, seed=None):
        if seed:
            np.random.seed(seed)

        class_a = np.array([np.random.normal(mean, sigma, self.n_a) for mean, sigma in zip(self.m_a, self.sigma_a)])
        class_b = np.array([np.random.normal(mean, sigma, self.n_a) for mean, sigma in zip(self.m_b, self.sigma_b)])
        targets = np.concatenate((np.ones(self.n_a), np.ones(self.n_b) * -1), axis=0).reshape(1, -1)

        patterns = np.concatenate((class_a, class_b), axis=1)  # column-wise concatenation
        patterns, targets = shuffle_two_arrays(patterns, targets)

        return patterns, targets

    def random_2(self, seed=None):
        if seed:
            np.random.seed(seed)

        class_a = np.concatenate((
            np.array([np.random.normal(mean, sigma, self.n_a // 2) for mean, sigma in
                      zip((-self.m_a[0], self.m_a[1]), self.sigma_a)]),
            np.array([np.random.normal(mean, sigma, self.n_a - self.n_a // 2) for mean, sigma in
                      zip(self.m_a, self.sigma_a)])
        ), axis=1)
        class_b = np.array([np.random.normal(mean, sigma, self.n_a) for mean, sigma in zip(self.m_b, self.sigma_b)])
        targets = np.concatenate((np.ones(self.n_a), np.ones(self.n_b) * -1), axis=0).reshape(1, -1)

        patterns = np.concatenate((class_a, class_b), axis=1)  # column-wise concatenation
        patterns, targets = shuffle_two_arrays(patterns, targets)

        return patterns, targets


def animate_lienar_separator_for_2d_features(inputs, targets, name, weights, convergance_epoch, save_folder=None):
    # matplotlib.use("TkAgg")

    # Help method that converts a weight into x&y start&end point coordinate data
    xmin = inputs[0].min() - 1
    xmax = inputs[0].max() + 1
    ymin = inputs[1].min() - 1
    ymax = inputs[1].max() + 1

    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))

    # set up the plot labels
    plt.title(name)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    # determine feature 1 and feature 2 and plot them in different colors
    positive = np.where(targets.flatten() == 1)
    negative = np.where(targets.flatten() != 1)
    ax.scatter(inputs[0][positive].T, inputs[1][positive].T, color="blue")
    ax.scatter(inputs[0][negative].T, inputs[1][negative].T, color="red")

    # line setup
    line, = ax.plot([], [], lw=2, label="Epoch", color="green")

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line, line.figure.legend()

    def weight_to_line_data(w, xmin=xmin, xmax=xmax):
        x = np.array([xmin, xmax])
        y = (- w[0, 0] * x - w[0, 2]) / w[0, 1]
        return [x[0], x[1]], [y[0], y[1]]

    def animate(i):
        x, y = weight_to_line_data(weights[i])
        line.set_data(x, y)
        line.set_label(f"Epoch {i + 1}")
        if i == convergance_epoch:
            line.set_color("red")
        legend = plt.legend()
        return line, legend

    epochs = len(weights)
    fps = min(60, math.ceil(epochs / 7))
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=epochs, interval=10,
                                  blit=True)

    if save_folder is not None:
        ani.save(os.path.join(save_folder, name) + ".mp4", fps=fps, extra_args=['-vcodec', 'libx264'], dpi=300)
        print(os.path.abspath(os.path.realpath(os.path.join(save_folder, name) + ".mp4")))
    plt.close()
    # plt.show()


def scatter_plot_2d_features(inputs, targets, name, line_coefficients=None, save_folder=None, show_plot=True,
                             fmt=("bo", "rx")):
    # plt.figure(figsize=(9, 6))
    plt.title(name)

    positive = np.where(targets.flatten() == 1)
    negative = np.where(targets.flatten() != 1)

    plt.plot(inputs[0][positive].T, inputs[1][positive].T, fmt[0])
    plt.plot(inputs[0][negative].T, inputs[1][negative].T, fmt[1])

    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.xlim(plt.xlim())  # lock limit on the x axis
    plt.ylim(plt.ylim())  # lock limit on the y axis
    if line_coefficients is not None:
        x = np.linspace(inputs[0].min() - 1, inputs[0].max() + 1, 100)
        y = (- line_coefficients[0, 0] * x - line_coefficients[0, 2]) / line_coefficients[0, 1]
        plt.plot(x, y)

    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, name) + ".png", dpi=300)
    if show_plot:
        plt.show()


def graph_surface(function, rect, offset=0.5, width=512, height=512):
    """Creates a surface plot (visualize with plt.show).
    Code from: http://www.zemris.fer.hr/~ssegvic/du/src/data.py

    Arguments:
      function: surface to be plotted
      rect:     function domain provided as:
                ([x_min,y_min], [x_max,y_max])
      offset:   the level plotted as a contour plot

    Returns:
      None
    """

    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    # get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values, vmin=delta - maxval, vmax=delta + maxval)

    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])


def conf_mat_acc(cm):
    return np.trace(cm) / np.sum(cm)


def rbf(x, c, variance):
    if variance == 0:
        return 0

    return np.exp(- np.sum((x - c) ** 2) / (2 * variance)).item()


def normalize_vectors(data, vectors_in_rows=True):
    if vectors_in_rows:
        return (data.T / np.sqrt(np.sum(data ** 2, axis=1))).T
    else:
        return data / np.sqrt(np.sum(data ** 2, axis=0))
