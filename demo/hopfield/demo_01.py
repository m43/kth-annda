import numpy as np


import sys
sys.path.append('../../model/')
from hopfield import Hopfield

#from model.hopfield import Hopfield


def plus_one(binary_pattern):
    for i in range(len(binary_pattern)):
        if binary_pattern[i] == -1:
            binary_pattern[i] = +1
            break
        else:
            binary_pattern[i] = -1


# create/load binary (-1, 1) patterns (NumPy format)
x1 = np.array([-1, -1, +1, -1, +1, -1, -1, +1])
x2 = np.array([-1, -1, -1, -1, -1, +1, -1, -1])
x3 = np.array([-1, +1, +1, -1, -1, +1, -1, +1])
patterns = np.array([x1, x2, x3])
pattern_set = {tuple(x1), tuple(x2), tuple(x3)}

# create an instance of the model
my_model = Hopfield(8)  # needs to have as much neurons as each pattern has features

# learn the patterns
my_model.learn_patterns(patterns, scaling=False)

# check for stable points
test_pattern = np.array([-1, -1, -1, -1, -1, -1, -1, -1])
stable_states = set()
state_count = dict()
unstable_count = 0
for i in range(2 ** 8):
    # print(test_pattern)
    my_model.set_state(test_pattern)
    stable_state = my_model.update_automatically()
    if stable_state is not None:
        stable_state = tuple(stable_state)
        if stable_state in stable_states:
            state_count[stable_state] = state_count[stable_state] + 1
        else:
            stable_states.add(stable_state)
            state_count[stable_state] = 1
    else:
        unstable_count += 1
    plus_one(test_pattern)
print(f'{len(stable_states)} stable states (attractors)  found, they are the following:')
for stable_state in stable_states:
    print(f'\t{stable_state} - attracted {state_count[stable_state]} points', end='')
    if stable_state in pattern_set:
        print(' - a learned pattern')
    else:
        print()
print(f'Remaining {unstable_count} states are unstable (they cause oscillation when using batch learning).')
