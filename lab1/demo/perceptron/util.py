import math
import os

from model.delta_rule_perceptron import DeltaRulePerceptron
from model.perceptron_rule_perceptron import PerceptronRulePerceptron
from utils.util import scatter_plot_2d_features, animate_lienar_separator_for_2d_features


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


