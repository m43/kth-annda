import numpy as np
from utils.util import horse

from model.rbf import Rbf

my_model = Rbf(5)
test_list = [1, 2, 3]

test_output_columns = None
for single_input in test_list:
    if test_output_columns is not None:
        test_output_columns = np.concatenate([test_output_columns, my_model.calculate_hidden_output(single_input)],
                                             axis=1)
    else:
        test_output_columns = my_model.calculate_hidden_output(single_input)

test_matrix = my_model.calculate_hidden_matrix(test_list)

for step, row in enumerate(test_matrix):
    if not (row == test_output_columns.T[step]).all():
        raise RuntimeError("Hidden matrix calculation not working properly!")
    else:
        print('All good!')

test_input = 0.5
expected_output = 2
print(f'For input {test_input} the output is currently: {my_model.forward_pass(test_input)}')
print('Before delta learning on the test input the weights were:\n', my_model.weights)
my_model.delta_training_step(test_input, expected_output, 0.01)
print('After delta learning on the test input the weights were:\n', my_model.weights)
print(f'For input {test_input} the output is currently: {my_model.forward_pass(test_input)}')

test_inputs = [0.5, -0.5, 1, -0.75, 2]
expected_outputs = [2, -2, 2, -2, 2]
print(f'For input {test_inputs} the output is currently: {[my_model.forward_pass(i) for i in test_inputs]}')
print('Before least square learning on the test input the weights were:\n', my_model.weights)
my_model.least_squares_training(test_inputs, expected_outputs)
print('After least square learning on the test input the weights were:\n', my_model.weights)
print(f'For input {test_inputs} the output is currently: {[my_model.forward_pass(i) for i in test_inputs]}')