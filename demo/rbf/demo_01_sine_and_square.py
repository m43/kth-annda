import math
import statistics
import matplotlib.pyplot as plt
from model.rbf import Rbf

NUMBER_OF_NODES = 7

inputs = [i / 100.0 for i in range(0, int(math.pi * 100), 10)]
sine_targets = [math.sin(2 * value) for value in inputs]
square_targets = [1 if math.sin(2 * value) >= 0 else -1 for value in inputs]

test_inputs = [i / 100.0 for i in range(5, int(math.pi * 100 + 5), 10)]
test_sine_targets = [math.sin(2 * value) for value in test_inputs]
test_square_targets = [1 if math.sin(2 * value) >= 0 else -1 for value in test_inputs]

# print(inputs)
# print(sine_targets)
# print(square_targets)

sine_rbf = Rbf(NUMBER_OF_NODES)
square_rbf = Rbf(NUMBER_OF_NODES)

sine_rbf.least_squares_training(inputs, sine_targets)
square_rbf.least_squares_training(inputs, square_targets)

# sine analysis
sine_outputs = [sine_rbf.forward_pass(value) for value in test_inputs]
sine_mae = statistics.mean(
    [math.fabs(sine_outputs[i] - sine_targets[i]) for i in range(len(sine_targets))])
print(f'Target values of sine wave: {test_sine_targets}')
print(f'Actual output: {sine_outputs}')
print(f'MAE is: {sine_mae}\n')

plt.title(f'Sine function approximation, n = {NUMBER_OF_NODES}')
plt.plot(inputs, sine_targets, label='true values')
plt.plot(inputs, sine_outputs, label='prediction values')
plt.xlabel('x')
plt.ylabel('sin(2x)')
plt.legend()
plt.show()

# square analysis
square_outputs = [square_rbf.forward_pass(value) for value in test_inputs]
corrected_square_outputs = [1 if square_output >= 0 else -1 for square_output in square_outputs]
square_mae = statistics.mean(
    [math.fabs(square_outputs[i] - square_targets[i]) for i in range(len(square_targets))])
corrected_square_mae = statistics.mean(
    [math.fabs(corrected_square_outputs[i] - square_targets[i]) for i in range(len(square_targets))])
print(f'Target values of sine wave: {test_square_targets}')
print(f'Actual output: {square_outputs}')
print(f'Corrected output: {corrected_square_outputs}')
print(f'MAE is: {square_mae}')
print(f'Corrected MAE is: {corrected_square_mae}\n')

plt.title(f'Square function approximation, n = {NUMBER_OF_NODES}')
plt.plot(inputs, square_targets, label='true values')
plt.plot(inputs, square_outputs, label='prediction values')
plt.xlabel('x')
plt.ylabel('square(2x)')
plt.legend()
plt.show()
