import numpy as np

from model.hopfield import Hopfield

np.set_printoptions(precision=3)  # limit NumPy printing to 3 decimal points


def my_callback(current_state, step):
    if step % 100 == 0:
        print(f'\tCurrently in step {step} of updating; the state is: {current_state}')


"""
This short script demonstrates how to use our implementation of the Hopfield network.
"""

# create/load binary (-1, 1) patterns (NumPy format)
x1 = np.array([-1, -1, +1, -1, +1, -1, -1, +1])
x2 = np.array([-1, -1, -1, -1, -1, +1, -1, -1])
x3 = np.array([-1, +1, +1, -1, -1, +1, -1, +1])
patterns = np.array([x1, x2, x3])

# create an instance of the model
my_model = Hopfield(8, debug_mode=True)  # needs to have as much neurons as each pattern has features

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
    print(f'\tIs this the state we were looking for? {(pattern == new_state).all()}!\n')

# check if learned patterns are stable states using asynchronous updating
print('Checking if learned patterns are stable states (asynchronous):')
for pattern in patterns:
    print(f'\tSetting state to pattern: {pattern}')
    my_model.set_state(pattern)
    new_state = my_model.update_step(batch=False)
    print(f'\tState after asynchronous update: {new_state}!')
    print(f'\tIs this the state we were looking for? {(pattern == new_state).all()}!\n')

# create patterns with errors
x1_e = np.array([+1, -1, +1, -1, +1, -1, -1, +1])  # 1 bit error
x2_e = np.array([+1, +1, -1, -1, -1, +1, -1, -1])  # 2 bit error
x3_e = np.array([+1, +1, +1, -1, +1, +1, -1, +1])  # 2 bit error
error_patterns = np.array([x1_e, x2_e, x3_e])

# check patterns with errors using synchronous updating
print('Checking network\'s capability of handling wrong inputs (synchronous):')
for pattern, error_pattern in zip(patterns, error_patterns):
    print(f'\tSetting state to pattern: {error_pattern}')
    my_model.set_state(error_pattern)
    new_state = my_model.update_automatically(step_callback=my_callback)
    print(f'\tState after synchronous updates: {new_state}!')
    print(f'\tIs this the state we were looking for? {(pattern == new_state).all()}!\n')

# check patterns with errors using asynchronous updating
print('Checking network\'s capability of handling wrong inputs (asynchronous):')
for pattern, error_pattern in zip(patterns, error_patterns):
    print(f'\tSetting state to pattern: {error_pattern}')
    my_model.set_state(error_pattern)
    new_state = my_model.update_automatically(batch=False, step_callback=my_callback)
    print(f'\tState after synchronous updates: {new_state}!')
    print(f'\tIs this the state we were looking for? {(pattern == new_state).all()}!\n')
