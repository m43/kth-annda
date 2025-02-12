import matplotlib.pyplot as plt
import numpy as np
import os
import statistics
import tqdm

from model.hopfield import Hopfield


def generate_random_from_prototype(prototype, amount):
    patterns = []
    patterns_set = set()
    for i in range(amount):
        while True:
            new_pattern = np.copy(prototype)
            np.random.shuffle(new_pattern)
            new_pattern_tuple = tuple(new_pattern)
            # make sure that the pattern doesn't already exist
            if new_pattern_tuple not in patterns_set:
                patterns_set.add(new_pattern_tuple)
                patterns.append(new_pattern)
                break

    return np.array(patterns)


BATCH = False
SELF_CONNECTIONS = False
NUMBER_OF_TESTS = 10
MAX_NUMBER_OF_PATTERNS = 10000
STEP = 1
DIMENSIONS = 100
SPARSITY = 0.1
OUTPUT_BIAS = 0.5
OUTPUT_SCALING = 0.5
SAVE_FOLDER = f'sparse_patterns_no_self_connections/sparsity={SPARSITY:.2f}'

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

biases = [0.05]
# biases = [0, 0.01, 0.1, 0.3, 0.5]
# biases = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
# biases = [0, -0.001, -0.005, -0.01, -0.05, -0.1, -0.3, -0.5, -0.7]
# biases = [0.0001, 0.0005]
# biases = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
# biases = [0.3, 0.7, 0.9]


# generate sparse pattern prototype
ones = int(DIMENSIONS * SPARSITY)
prototype_pattern = np.zeros(100)
prototype_pattern[0:ones] = [1 for i in range(ones)]

# create model
my_model = Hopfield(DIMENSIONS)

# data dictionary
bias_data = dict()
# test
print('Testing biases...')
for bias in tqdm.tqdm(biases):
    # create dictionary with data for current bias
    stabilities = dict()
    for i in range(0, MAX_NUMBER_OF_PATTERNS, STEP):
        stabilities[i + 1] = []

    for i in range(NUMBER_OF_TESTS):
        my_patterns = generate_random_from_prototype(prototype_pattern, MAX_NUMBER_OF_PATTERNS)
        for j in range(0, MAX_NUMBER_OF_PATTERNS, STEP):
            my_model.learn_patterns(my_patterns[0:j + 1], imbalance=SPARSITY, self_connections=SELF_CONNECTIONS)
            stability = 0.0
            for pattern in my_patterns[0:j + 1]:
                my_model.set_state(pattern)
                after_update = my_model.update_step(batch=BATCH, bias=bias, output_bias=OUTPUT_BIAS,
                                                    output_scaling=OUTPUT_SCALING)
                stability += 1 if (after_update == pattern).all() else 0
            stability = stability / (j + 1)
            stabilities[j + 1].append(stability)

    # save results of current bias to data dictionary
    bias_data[bias] = stabilities

# create plots and save results
x_axis = [i + 1 for i in range(0, MAX_NUMBER_OF_PATTERNS, STEP)]
for bias in biases:
    stabilities = bias_data[bias]
    save_path = f'{SAVE_FOLDER}/sparsity={SPARSITY}_bias={bias:3.5f}'

    # save results
    number_of_patterns = [i + 1 for i in range(0, MAX_NUMBER_OF_PATTERNS, STEP)]
    means = [statistics.mean(stabilities[i + 1]) for i in range(0, MAX_NUMBER_OF_PATTERNS, STEP)]
    standard_deviations = [statistics.stdev(stabilities[i + 1]) for i in range(0, MAX_NUMBER_OF_PATTERNS, STEP)]
    with open(save_path + '.csv', 'w') as f:
        f.write('bias, n, mean, std\n')
        for number, mean, std in zip(number_of_patterns, means, standard_deviations):
            f.write(f'{bias}, {number}, {mean}, {std}\n')

    # create plots
    y_axis = [statistics.mean(stabilities[i + 1]) for i in range(0, MAX_NUMBER_OF_PATTERNS, STEP)]
    plt.plot(x_axis, y_axis)
    plt.title(f'SPARSITY={SPARSITY}, BIAS={bias:.2f}, NUMBER_OF_TESTS={NUMBER_OF_TESTS}')
    plt.xlabel('number of patterns network tried to learn')
    plt.ylabel('average percentage of learned patterns')
    plt.savefig(save_path + '.png')
    plt.close()
