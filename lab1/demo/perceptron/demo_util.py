import math
import os

import numpy as np

from model.delta_rule_perceptron import DeltaRulePerceptron
from model.perceptron_rule_perceptron import PerceptronRulePerceptron
from utils.util import scatter_plot_2d_features, TwoClassDatasetGenerator, animate_lienar_separator_for_2d_features


def perpare_reproducable_separable_dataset():
    return TwoClassDatasetGenerator(
        m_a=(1.5, 1.5), n_a=100, sigma_a=(0.5, 0.5),
        m_b=(-1.5, -1.5), n_b=100, sigma_b=(0.5, 0.5)
    ).random_1(seed=72)


def perpare_reproducable_separable_dataset_impossible_with_no_bias():
    TwoClassDatasetGenerator(
        m_a=(1, 1), n_a=100, sigma_a=(0.5, 1),
        m_b=(4, 4), n_b=100, sigma_b=(0.6, 0.3)
    ).random_1(seed=72)


def perpare_reproducable_inseparable_dataset_1():
    return TwoClassDatasetGenerator(
        m_a=(1.8, 1.8), n_a=100, sigma_a=(1, 1),
        m_b=(0, 0), n_b=100, sigma_b=(0.5, 0.7)
    ).random_1(seed=72)


def perpare_reproducable_inseparable_dataset_2():
    return TwoClassDatasetGenerator(
        m_a=(1.0, 0.3), n_a=100, sigma_a=(0.2, 0.2),
        m_b=(0.0, -0.1), n_b=100, sigma_b=(0.3, 0.3)
    ).random_2(seed=72)


def perceptron_learning_demo(inputs, targets, name, debug, save_folder, max_iter, eta):
    if debug:
        print(f"{name} started")

    perceptron = PerceptronRulePerceptron(inputs, targets, debug, os.path.join(save_folder, name))
    if debug:
        print(f"Before training - Acc: {perceptron.eval(inputs, targets)}")
        scatter_plot_2d_features(inputs, targets, name + "_START", perceptron.W, save_folder)

    try:
        weights_per_epoch, acc, convergence_epoch = perceptron.train(inputs, targets, eta, max_iter=max_iter,
                                                                     shuffle=True)
    except RuntimeError:
        print("RUNTIME ERROR in train...")
        return math.nan, math.nan

    if debug:
        print(f"After training - Acc: {acc} CEpoch: {convergence_epoch}")
        scatter_plot_2d_features(inputs, targets, name + "_END", perceptron.W, save_folder)
        animate_lienar_separator_for_2d_features(
            inputs, targets, name + "_animation", weights_per_epoch, convergence_epoch, save_folder)
        print()

    return acc, convergence_epoch


def delta_rule_learning_demo(inputs, targets, name, debug, save_folder, max_iter, eta, delta_n, batch_size, bias,
                             plots_with_debug=True, confusion_matrix=False):
    if debug:
        print(f"{name} started")

    perceptron = DeltaRulePerceptron(inputs, targets, debug, os.path.join(save_folder, name), bias=bias)
    if debug:
        perceptron.debug = False
        acc, loss = perceptron.eval(inputs, targets)
        perceptron.debug = debug
        print(f"Before training - Acc: {acc} Loss: {loss}")
        if plots_with_debug:
            scatter_plot_2d_features(inputs, targets, name + "_START", perceptron.W, save_folder)

    try:
        weights_per_epoch, acc, loss, convergence_epoch = perceptron.train(inputs, targets, eta, max_iter, batch_size,
                                                                           shuffle=True, stop_after=delta_n)
    except RuntimeError:
        print("RUNTIME ERROR in train...")
        return math.nan, math.nan, math.nan

    if debug:
        print(f"After training - Acc: {acc} Loss: {loss} CEpoch: {convergence_epoch}")
        if plots_with_debug:
            scatter_plot_2d_features(inputs, targets, name + "_END", perceptron.W, save_folder)
            animate_lienar_separator_for_2d_features(
                inputs, targets, name + "_animation", weights_per_epoch, convergence_epoch, save_folder)
        print()

    if confusion_matrix:
        perceptron.debug = True
        acc, loss = perceptron.eval(inputs, targets)
        perceptron.debug = debug

    return acc, loss, convergence_epoch


def print_results_as_table(results, keys):
    print("", end="\t")
    for k, v in results.items():
        print(k, end="\t")
    print()
    for key in keys:
        print(f"{key}_mean", end="\t")
        for k, v in results.items():
            print(v[key][0], end="\t")
        print()
        print(f"{key}_std", end="\t")
        for k, v in results.items():
            print(v[key][1], end="\t")
        print()


def two_class_conf_mat_metrics(cm):
    tp, tn, fp, fn = cm[0, 0], cm[1, 1], cm[0, 1], cm[1, 0]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return accuracy, sensitivity, specificity, precision, recall


def sample_two_class_dataset(inputs, targets, n_a, n_b):
    idx_1 = []
    for i, t in enumerate(targets[0]):
        if t == 1 and n_a > 0:
            idx_1.append(i)
            n_a -= 1
        if t != 1 and n_b > 0:
            idx_1.append(i)
            n_b -= 1
    return np.delete(inputs, idx_1, axis=1), np.delete(targets, idx_1, axis=1)