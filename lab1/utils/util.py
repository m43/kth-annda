import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

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


def plot_metric(data, name, save,  *more_data, point=None):
    plt.figure(figsize=(9, 6))
    x = np.arange(len(data))
    plt.title(name)
    plt.plot(x, data, color="green")
    for d in more_data:
        plt.plot(x, d)
    if point is not None:
        plt.scatter(point[0], point[1], color="red")
    if save:
        plt.savefig(name)
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

    def random(self, seed=None):
        if seed:
            np.random.seed(seed)

        class_a = np.array([np.random.normal(mean, sigma, self.n_a) for mean, sigma in zip(self.m_a, self.sigma_a)])
        class_b = np.array([np.random.normal(mean, sigma, self.n_a) for mean, sigma in zip(self.m_b, self.sigma_b)])
        targets = np.concatenate((np.ones(self.n_a), np.ones(self.n_b) * -1), axis=0).reshape(1, -1)

        patterns = np.concatenate((class_a, class_b), axis=1)  # column-wise concatenation
        patterns, targets = shuffle_two_arrays(patterns, targets)

        return patterns, targets


def scatter_plot_2d_features(inputs, targets: np.array, name, save_folder=None, line_coefficients=None):
    plt.figure(figsize=(9, 6))
    plt.title(name)

    positive = np.where(targets.flatten() == 1)
    negative = np.where(targets.flatten() != 1)

    plt.plot(inputs[0][positive].T, inputs[1][positive].T, "bo")
    plt.plot(inputs[0][negative].T, inputs[1][negative].T, "rx")

    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.xlim(plt.xlim())  # lock limit on the x axis
    plt.ylim(plt.ylim())  # lock limit on the y axis
    if line_coefficients is not None:
        x = np.linspace(-5, 5, 100)
        y = (- line_coefficients[0, 0] * x - line_coefficients[0, 2]) / line_coefficients[0, 1]
        plt.plot(x, y)

    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, name) + ".png")
    plt.show()
