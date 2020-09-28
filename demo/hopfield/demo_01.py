import numpy as np

from model.hopfield import Hopfield


def plus_one(bipolar_pattern):
    for i in range(len(bipolar_pattern)):
        if bipolar_pattern[i] == -1:
            bipolar_pattern[i] = +1
            break
        else:
            bipolar_pattern[i] = -1


# create/load binary (-1, 1) patterns (NumPy format)
x1 = np.array([-1, -1, +1, -1, +1, -1, -1, +1])
x2 = np.array([-1, -1, -1, -1, -1, +1, -1, -1])
x3 = np.array([-1, +1, +1, -1, -1, +1, -1, +1])
patterns = np.array([x1, x2, x3])
pattern_set = {tuple(x1), tuple(x2), tuple(x3)}

# create an instance of the model
my_model = Hopfield(8)  # needs to have as much neurons as each pattern has features

x1d = x1.copy()
x1d[[0]] *= -1
x2d = x2.copy()
x2d[[0, 1]] *= -1
x3d = x3.copy()
x3d[[0, 4]] *= -1
for diagonal in (True, False):
    my_model.learn_patterns(patterns, scaling=False, self_connections=diagonal)
    for batch in (True, False):
        for x, xd, name in ((x1, x1d, "x1"), (x2, x2d, "x2"), (x3, x3d, "x3")):
            print(f"*** Diagonal: {diagonal} *** Batch: {batch} ***")
            my_model.set_state(xd)
            stable_state = my_model.update_automatically(batch=batch)
            converged = stable_state is not None and (stable_state == x).all()
            print(f"{'YES' if converged else 'NO'}: Distorted {name} converged {'' if converged else 'un'}successfully")
            print(f"x:{x} xd:{xd} xconverged:{stable_state}")
        print()

# # learn the patterns
# my_model.learn_patterns(patterns, scaling=False)

for diagonal in (True, False):
    my_model.learn_patterns(patterns, scaling=False, self_connections=diagonal)
    for batch in (True, False):
        # check for stable points
        test_pattern = np.array([-1, -1, -1, -1, -1, -1, -1, -1])
        stable_states = set()
        state_count = dict()
        unstable_count = 0
        for i in range(2 ** 8):
            my_model.set_state(test_pattern)
            stable_state = my_model.update_automatically(batch=batch)
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
        print(f'Batch:{batch} Diagonal:{diagonal} ::: '
              f'{len(stable_states)} stable states (attractors) found, they are the following:')
        for stable_state in sorted(stable_states):
            print(f'\t{np.array(stable_state, dtype=int)} - attracted {state_count[stable_state]} points', end='')
            if stable_state in pattern_set:
                print(' - a learned pattern')
            else:
                print()
        print(f'Remaining {unstable_count} states are unstable (they cause oscillation when using batch learning).')
        print()
