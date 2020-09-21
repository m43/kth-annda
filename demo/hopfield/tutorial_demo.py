import numpy as np

from model.hopfield import Hopfield

np.set_printoptions(precision=3)  # limit NumPy printing to 3 decimal points

"""
This short script demonstrates how to use our implementation of the Hopfield network.
"""

# create/load binary (-1, 1) patterns (NumPy format)
x1 = np.array([+1, -1, +1, -1, +1, -1, -1, +1])
x2 = np.array([+1, +1, -1, -1, -1, +1, -1, -1])
x3 = np.array([+1, +1, +1, -1, +1, +1, -1, +1])
patterns = np.array([x1, x2, x3])

# create an instance of the model
my_model = Hopfield(8)  # needs to have as much neurons as each pattern has features

# learn the patterns
print(f'Learning model on the following patterns:\n{patterns}\n')
my_model.learn_patterns(patterns, scaling=False)

# print weights
print(f'Model learned patterns, the weight matrix is as follows:\n{my_model.weights}\n')

# check if learned patterns are stable states using synchronous updating
print('Checking if learned patterns are stable states (synchronous):')
for pattern in patterns:
    print(f'\tSetting state to pattern: {pattern}')
    my_model.set_state(pattern)
    new_state = my_model.update_step()
    print(f'\tState after synchronous update: {new_state}!')
    print(f'\tState is stable? {(pattern == new_state).all()}!\n')

# check if learned patterns are stable states using asynchronous updating
print('Checking if learned patterns are stable states (asynchronous):')
for pattern in patterns:
    print(f'\tSetting state to pattern: {pattern}')
    my_model.set_state(pattern)
    new_state = my_model.update_step(batch=False)
    print(f'\tState after asynchronous update: {new_state}!')
    print(f'\tState is stable? {(pattern == new_state).all()}!\n')
