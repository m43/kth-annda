from pathlib import Path

import numpy as np

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
    return np.sum((targets - outputs) ** 2) / len(outputs)


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
