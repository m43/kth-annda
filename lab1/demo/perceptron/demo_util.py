import math
import os

from model.delta_rule_perceptron import DeltaRulePerceptron
from model.perceptron_rule_perceptron import PerceptronRulePerceptron
from utils.util import scatter_plot_2d_features, TwoClassDatasetGenerator, animate_lienar_separator_for_2d_features


def perpare_reproducable_separable_dataset():
    return TwoClassDatasetGenerator(
        m_a=(1.5, 1.5), n_a=100, sigma_a=(0.5, 0.5),
        m_b=(-1.5, -1.5), n_b=100, sigma_b=(0.5, 0.5)
    ).random(seed=72)


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


def delta_rule_learning_demo(inputs, targets, name, debug, save_folder, max_iter, eta, delta_n, batch_size, bias):
    if debug:
        print(f"{name} started")

    perceptron = DeltaRulePerceptron(inputs, targets, debug, os.path.join(save_folder, name), bias=bias)
    if debug:
        acc, loss = perceptron.eval(inputs, targets)
        print(f"Before training - Acc: {acc} Loss: {loss}")
        scatter_plot_2d_features(inputs, targets, name + "_START", perceptron.W, save_folder)

    try:
        weights_per_epoch, acc, loss, convergence_epoch = perceptron.train(inputs, targets, eta, max_iter, batch_size,
                                                                           shuffle=True, stop_after=delta_n)
    except RuntimeError:
        print("RUNTIME ERROR in train...")
        return math.nan, math.nan, math.nan

    if debug:
        print(f"After training - Acc: {acc} Loss: {loss} CEpoch: {convergence_epoch}")
        scatter_plot_2d_features(inputs, targets, name + "_END", perceptron.W, save_folder)
        animate_lienar_separator_for_2d_features(
            inputs, targets, name + "_animation", weights_per_epoch, convergence_epoch, save_folder)
        print()

    return acc, loss, convergence_epoch


def print_results_as_table(results, keys):
    for k, v in results.items():
        print(k, end="\t")
    print()
    for key in keys:
        for k, v in results.items():
            print(v[key][0], end="\t")
        print()
        for k, v in results.items():
            print(v[key][1], end="\t")
        print()
