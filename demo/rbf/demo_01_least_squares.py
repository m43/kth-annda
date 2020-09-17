import math
import statistics
import os
import matplotlib.pyplot as plt
from model.rbf import Rbf

MAX_NODES = 32
# 0 uses default variance for the uniform RBF initialization (a square root of the distance between RBF centers)
VARIANCES = [0]
# [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

# training inputs and outputs
inputs = [i / 100.0 for i in range(0, int(2 * math.pi * 100), 10)]
sine_targets = [math.sin(2 * value) for value in inputs]
square_targets = [1 if math.sin(2 * value) >= 0 else -1 for value in inputs]

# test inputs and outputs
test_inputs = [i / 100.0 for i in range(5, int(2 * math.pi * 100), 10)]
test_sine_targets = [math.sin(2 * value) for value in test_inputs]
test_square_targets = [1 if math.sin(2 * value) >= 0 else -1 for value in test_inputs]

# print(inputs)
# print(sine_targets)
# print(square_targets)

for variance in VARIANCES:

    save_folder = f'results_01/uniform_var={variance}/'

    if variance == 0:
        save_folder = f'results_01/uniform_var=root_square_distance/'

    sine_errors = []
    square_errors = []
    corrected_square_errors = []

    # create results_01 folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for number_of_nodes in range(1, MAX_NODES + 1):
        # model initialization
        sine_rbf = Rbf(number_of_nodes, rbf_init='uniform', rbf_init_data=[0, 2 * math.pi, variance])
        square_rbf = Rbf(number_of_nodes, rbf_init='uniform', rbf_init_data=[0, 2 * math.pi, variance])

        # model training
        sine_rbf.least_squares_training(inputs, sine_targets)
        square_rbf.least_squares_training(inputs, square_targets)

        # sine analysis
        training_sine_outputs = [sine_rbf.forward_pass(value) for value in inputs]
        sine_outputs = [sine_rbf.forward_pass(value) for value in test_inputs]
        sine_mae = statistics.mean(
            [math.fabs(sine_outputs[i] - test_sine_targets[i]) for i in range(len(test_sine_targets))])
        sine_errors.append(sine_mae)
        # print(f'Target values of sine wave: {test_sine_targets}')
        # print(f'Actual output: {sine_outputs}')
        # print(f'MAE is: {sine_mae}\n')

        # plot sin(2x) test prediction
        plt.title(f'Sine function approximation, n = {number_of_nodes}')
        plt.plot(test_inputs, test_sine_targets, label='true test values')
        plt.plot(test_inputs, sine_outputs, label='prediction test values')
        plt.xlabel('x')
        plt.ylabel('sin(2x)')
        plt.legend()
        plt.savefig(fname=f'{save_folder}/sin(2x)_nodes={number_of_nodes}', dpi=300)
        plt.close()

        # plot sin(2x) training prediction
        plt.title(f'Sine function approximation, n = {number_of_nodes}')
        plt.plot(test_inputs, test_sine_targets, label='true training values')
        plt.plot(test_inputs, sine_outputs, label='prediction training values')
        plt.xlabel('x')
        plt.ylabel('sin(2x)')
        plt.legend()
        plt.savefig(fname=f'{save_folder}/training_sin(2x)_nodes={number_of_nodes}', dpi=300)
        plt.close()

        # square analysis
        training_square_outputs = [square_rbf.forward_pass(value) for value in inputs]
        square_outputs = [square_rbf.forward_pass(value) for value in test_inputs]
        corrected_square_outputs = [1 if square_output >= 0 else -1 for square_output in square_outputs]
        square_mae = statistics.mean(
            [math.fabs(square_outputs[i] - test_square_targets[i]) for i in range(len(test_square_targets))])
        square_errors.append(square_mae)
        corrected_square_mae = statistics.mean(
            [math.fabs(corrected_square_outputs[i] - test_square_targets[i]) for i in range(len(test_square_targets))])
        corrected_square_errors.append(corrected_square_mae)
        # print(f'Target values of sine wave: {test_square_targets}')
        # print(f'Actual output: {square_outputs}')
        # print(f'Corrected output: {corrected_square_outputs}')
        # print(f'MAE is: {square_mae}')
        # print(f'Corrected MAE is: {corrected_square_mae}\n')

        # plot square(2x) test prediction
        plt.title(f'Square function approximation, n = {number_of_nodes}')
        plt.plot(test_inputs, test_square_targets, label='true test values')
        plt.plot(test_inputs, square_outputs, label='prediction test values')
        plt.xlabel('x')
        plt.ylabel('square(2x)')
        plt.legend()
        plt.savefig(fname=f'{save_folder}/square(2x)_nodes={number_of_nodes}', dpi=300)
        plt.close()

        # plot square(2x) training prediction
        plt.title(f'Square function approximation, n = {number_of_nodes}')
        plt.plot(inputs, square_targets, label='true training values')
        plt.plot(inputs, training_square_outputs, label='prediction training values')
        plt.xlabel('x')
        plt.ylabel('square(2x)')
        plt.legend()
        plt.savefig(fname=f'{save_folder}/training_square(2x)_nodes={number_of_nodes}', dpi=300)
        plt.close()

    # sine node performance plot
    plt.title(f'sin(2x) MAE vs. number of nodes')
    plt.plot([i for i in range(1, MAX_NODES + 1)], sine_errors)
    plt.xlabel('number of nodes')
    plt.ylabel('test MAE')
    plt.savefig(fname=f'{save_folder}/sine(2x)_node_performance', dpi=300)
    plt.close()

    # square node performance plot
    plt.title(f'square(2x) MAE vs. number of nodes')
    plt.plot([i for i in range(1, MAX_NODES + 1)], square_errors, label='real outputs')
    plt.plot([i for i in range(1, MAX_NODES + 1)], corrected_square_errors, label='corrected outputs')
    plt.xlabel('number of nodes')
    plt.ylabel('test MAE')
    plt.legend()
    plt.savefig(fname=f'{save_folder}/square(2x)_node_performance', dpi=300)
    plt.close()

    # save errors to .csv
    with open(f'{save_folder}/sin(2x)_errors.csv', 'a') as f:
        for nodes, error in enumerate(sine_errors, 1):
            f.write(f'{nodes}, {error}\n')
    with open(f'{save_folder}/square(2x)_errors.csv', 'a') as f:
        for nodes, error in enumerate(square_errors, 1):
            f.write(f'{nodes}, {error}\n')
    with open(f'{save_folder}/square(2x)_corrected_errors.csv', 'a') as f:
        for nodes, error in enumerate(corrected_square_errors, 1):
            f.write(f'{nodes}, {error}\n')
