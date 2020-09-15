import math
import matplotlib.pyplot as plt
from model.rbf import Rbf

NUMBER_OF_NODES = 8

inputs = [i / 100.0 for i in range(0, int(math.pi * 100), 10)]
sine_targets = [math.sin(2 * value) for value in inputs]
square_targets = [1 if math.sin(2 * value) >= 0 else -1 for value in inputs]

print(inputs)
print(sine_targets)
print(square_targets)

sine_rbf = Rbf(NUMBER_OF_NODES)
square_rbf = Rbf(NUMBER_OF_NODES)

sine_rbf.least_squares_training(inputs, sine_targets)
square_rbf.least_squares_training(inputs, square_targets)

sine_outputs = [sine_rbf.forward_pass(value) for value in inputs]
plt.title('Sine function approximation')
print(f'Target values of sine wave: {sine_targets}')
print(f'Actual output: {sine_outputs}')
plt.plot(inputs, sine_targets, label='true values')
plt.plot(inputs, sine_outputs, label='prediction values')
plt.show()

square_output = [square_rbf.forward_pass(value) for value in inputs]
plt.title('Square function approximation')
print(f'Target values of sine wave: {square_targets}')
print(f'Actual output: {square_output}')
plt.plot(inputs, square_targets, label='true values')
plt.plot(inputs, square_output, label='prediction values')
plt.show()
