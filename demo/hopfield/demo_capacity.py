import datetime
import numpy as np
import os

from model.hopfield import Hopfield
from utils.util import ensure_dir

picture_size = 1024
save_folder = '../../imgs/hopfield/demo_capacity/'
batch = True
debug = False
diagonal = False
# distortion_factors = [0, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
distortion_factors = [0, 0.001, 0.01, 0.1]
n_runs = 10

ensure_dir(save_folder)

# load pictures into a numpy float 2D array, 1 row per picture
with open('../../datasets/pict.dat', 'r') as source:
    data = source.read().strip('\n')
    data = data.split(',')
    pics = np.array([np.array(data[i:i + picture_size]).astype(np.float) for i in range(0, len(data), 1024)])
print(f'Successfully loaded {len(pics)} pictures.\n')

# create model
hop = Hopfield(picture_size, debug_mode=debug)

np.random.seed(72)
patterns_to_learn = []
# patterns_to_learn = [pics[:i] for i in range(1, 9 + 1)]
# patterns_to_learn += [pics[[0,1,2,4]], pics[[0,1,2,5]], pics[[0,1,2,6]], pics[[0,1,2,7]], pics[[0,1,2,5,6]]]
random_patterns_to_learn = []
# for i in list(range(1, 61, 10)) + list(range(61, 91 + 1)) + [150, 200, 250]:
#     random_patterns_to_learn.append(np.array([np.where(np.random.normal(size=1024) >= 0, 1, -1) for _ in range(i)]))
# for i in list(range(1, 181 + 1)):
#     random_patterns_to_learn.append(np.array([np.where(np.random.normal(size=1024) >= 0, 1, -1) for _ in range(i)]))
# for i in list(range(1, 21 + 1)) + [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
#     random_patterns_to_learn.append(
#         np.concatenate((pics[0:3], np.array([np.where(np.random.normal(size=1024) >= 0, 1, -1) for _ in range(i)]))))
biased_random_patterns_to_learn = []
all_biased_patterns = np.array([np.where(np.random.normal(0.5, size=1024) >= 0, 1, -1) for _ in range(301)])
# for i in list(range(1, 301 + 1, 10)):
#     random_patterns_to_learn.append(all_biased_patterns[0:i + 1])
for i in list(range(1, 21 + 1)):
    random_patterns_to_learn.append(all_biased_patterns[0:i + 1])


def distort_bipolar_pattern(pattern, n_of_distortions):
    result = pattern.copy()
    result[np.random.choice(range(len(result)), n_of_distortions, False).tolist()] *= -1
    return result


results = []
for patterns_name, collection_of_patterns in [("pict.dat", patterns_to_learn),
                                              ("random", random_patterns_to_learn),
                                              ("biased_random", biased_random_patterns_to_learn)]:
    for patterns in collection_of_patterns:
        current_results = []
        hop.learn_patterns(patterns, self_connections=diagonal)
        for distortion_factor in distortion_factors:
            n_distortions = int(distortion_factor * hop.number_of_neurons)
            success_count = 0
            for p in patterns:
                for _ in range(n_runs):
                    dp = distort_bipolar_pattern(p, n_distortions)
                    hop.set_state(dp)
                    if (p == hop.update_automatically(batch)).all():
                        success_count += 1
            print(f"{patterns_name}{len(patterns):03d}:::{distortion_factor}--{n_distortions}"
                  f":::{success_count}/{n_runs * len(patterns)}--{success_count / n_runs / len(patterns)}")
            current_results.append(success_count / n_runs / len(patterns))
        print()
        # results.append((f"{patterns_name}{len(patterns):3d}", current_results))
        results.append((f"{len(patterns)}", current_results))

with open(os.path.join(save_folder, f"results diag={diagonal} batch={batch} "
                                    f"n_runs={n_runs} time={datetime.datetime.now()}.csv"), "w") as csv:
    csv.write("learned patterns / distortion factor")
    for distortion_factor in distortion_factors:
        csv.write(f",{distortion_factor}")
    csv.write("\n")

    for name, current_results in results:
        csv.write(name)
        for distortion_factor, percentage in zip(distortion_factors, current_results):
            csv.write(f",{percentage}")
        csv.write("\n")
