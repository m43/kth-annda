import math
import statistics
import os
import matplotlib.pyplot as plt
from model.rbf import Rbf

MAX_NODES = 64
NOISE = False
SAVE_FOLDER = 'results/uniform_var=4_times_squared_distance/'

# create results folder
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

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

sine_errors = []
square_errors = []
corrected_square_errors = []

for number_of_nodes in range(1, MAX_NODES + 1):
    # model initialization
    sine_rbf = Rbf(number_of_nodes, rbf_init='uniform', rbf_init_data=[0, 2 * math.pi])
    square_rbf = Rbf(number_of_nodes, rbf_init='uniform', rbf_init_data=[0, 2 * math.pi])

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

    plt.title(f'Sine function approximation, n = {number_of_nodes}')
    plt.plot(test_inputs, test_sine_targets, label='true test values')
    plt.plot(test_inputs, sine_outputs, label='prediction test values')
    plt.xlabel('x')
    plt.ylabel('sin(2x)')
    plt.legend()
    plt.savefig(fname=f'{SAVE_FOLDER}/sin(2x)_nodes={number_of_nodes}', dpi=300)
    plt.close()

    plt.title(f'Sine function approximation, n = {number_of_nodes}')
    plt.plot(test_inputs, test_sine_targets, label='true training values')
    plt.plot(test_inputs, sine_outputs, label='prediction training values')
    plt.xlabel('x')
    plt.ylabel('sin(2x)')
    plt.legend()
    plt.savefig(fname=f'{SAVE_FOLDER}/training_sin(2x)_nodes={number_of_nodes}', dpi=300)
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

    plt.title(f'Square function approximation, n = {number_of_nodes}')
    plt.plot(test_inputs, test_square_targets, label='true test values')
    plt.plot(test_inputs, square_outputs, label='prediction test values')
    plt.xlabel('x')
    plt.ylabel('square(2x)')
    plt.legend()
    plt.savefig(fname=f'{SAVE_FOLDER}/square(2x)_nodes={number_of_nodes}', dpi=300)
    plt.close()

    plt.title(f'Square function approximation, n = {number_of_nodes}')
    plt.plot(inputs, square_targets, label='true training values')
    plt.plot(inputs, training_square_outputs, label='prediction training values')
    plt.xlabel('x')
    plt.ylabel('square(2x)')
    plt.legend()
    plt.savefig(fname=f'{SAVE_FOLDER}/training_square(2x)_nodes={number_of_nodes}', dpi=300)
    plt.close()

# sine node performance plot
plt.title(f'sin(2x) MAE vs. number of nodes')
plt.plot([i for i in range(1, MAX_NODES + 1)], sine_errors)
plt.xlabel('number of nodes')
plt.ylabel('test MAE')
plt.savefig(fname=f'{SAVE_FOLDER}/sine(2x)_node_performance', dpi=300)
plt.close()

# square node performance plot
plt.title(f'square(2x) MAE vs. number of nodes')
plt.plot([i for i in range(1, MAX_NODES + 1)], square_errors, label='real outputs')
plt.plot([i for i in range(1, MAX_NODES + 1)], corrected_square_errors, label='corrected outputs')
plt.xlabel('number of nodes')
plt.ylabel('test MAE')
plt.legend()
plt.savefig(fname=f'{SAVE_FOLDER}/square(2x)_node_performance', dpi=300)
plt.close()

# save errors to .csv
with open(f'{SAVE_FOLDER}/sin(2x)_errors.csv', 'a') as f:
    for nodes, error in enumerate(sine_errors, 1):
        f.write(f'{nodes}, {error}\n')
with open(f'{SAVE_FOLDER}/square(2x)_errors.csv', 'a') as f:
    for nodes, error in enumerate(square_errors, 1):
        f.write(f'{nodes}, {error}\n')
with open(f'{SAVE_FOLDER}/square(2x)_corrected_errors.csv', 'a') as f:
    for nodes, error in enumerate(corrected_square_errors, 1):
        f.write(f'{nodes}, {error}\n')
