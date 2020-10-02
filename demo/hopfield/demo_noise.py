import os

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../model/')
from hopfield import Hopfield

PICTURE_SIZE = 1024
SAVE_FOLDER = 'demo_02/pictures/'
BATCH = True
DEBUG = False

N_SAMPLE_ATTRACTORS = 1000


if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# load pictures into a numpy float 2D array, 1 row per picture
with open('../../datasets/pict.dat', 'r') as source:
    data = source.read().strip('\n')
    data = data.split(',')
    pics = np.array([np.array(data[i:i + PICTURE_SIZE]).astype(np.float) for i in range(0, len(data), 1024)])
print(f'Successfully loaded {len(pics)} pictures.\n')

# create model
my_model = Hopfield(PICTURE_SIZE, debug_mode=DEBUG)

# learn first three pictures
print("Learning pictures p1, p2, and p3.")
my_model.learn_patterns(pics[0:3])



# from copy import deepcopy

# test = deepcopy(pics[0])

# Function to generate noisy samples
def add_noise(pattern, proportion):
    noisy = pattern.copy()
    nflips = int(proportion * pattern.shape[0])
    indexes = np.random.choice(pattern.shape[0], nflips, False) # sample random indexes
    noisy[indexes] *= -1 # flip sign
    return noisy

# def distort_bipolar_pattern(pattern, proportion):
#     nflips = int(proportion * pattern.shape[0])
#     result = pattern.copy()
#     result[np.random.choice(range(len(result)), nflips, False).tolist()] *= -1
#     print(proportion, " - ", np.sum(result != pattern))
#     return result


# def find_most_similar_attractor(pattern):
#     best = (0, None)
#     for i in range(3):
#         sim = np.sum(pattern == pics[i]) / pics[i].shape[0]
#         if sim > best[0]:
#             best = (sim, i)
#     return best

print("generate noisy versions of p1, p2 and p3 and see how much noise can be removed")

NAVERAGE = 30
ls = [[], [], []]
iters = [[], [], []]
buff = []
buffer_callback = lambda x,_: buff.append(my_model.state)

for i in range(3):
    for j in np.arange(0.01, 1.01, 0.01):
        avg = 0
        avg_it = 0
        for _ in range(NAVERAGE):
            buff = [] # reset buffer
            
            noisy_p = add_noise(pics[i], j)
            my_model.set_state(noisy_p)
            state = my_model.update_automatically(batch=BATCH, step_callback=buffer_callback)
            
            retrived_percentage = 1-np.sum(state != pics[i]) / pics[i].shape[0]
            avg += retrived_percentage
            # avg += np.sum(noisy_p != pics[i]) / pics[i].shape[0]
            

            niter = len(buff)
            avg_it += niter
            # final_state = buff[-1]
            # attractor = find_most_similar_attractor(final_state)

        iters[i].append(avg_it/NAVERAGE)
        ls[i].append(avg/NAVERAGE)

plt.plot(list(range(len(ls[0]))), ls[0], "r-", label="p1")
plt.plot(list(range(len(ls[1]))), ls[1], "g-", label="p2")
plt.plot(list(range(len(ls[2]))), ls[2], "b-", label="p3")
plt.xlabel("percentage of noise")
plt.ylabel("retreived proportion of image")
plt.title(f'Distortion resistance tests averaged over {NAVERAGE} runs')
plt.legend()
# plt.savefig(fname=f'{SAVE_FOLDER}noise_distortion_01.eps', format='eps', bbox_inches='tight')
plt.savefig(fname=f'{SAVE_FOLDER}noise_distortion_01', bbox_inches='tight')
# plt.show()
plt.close()

plt.plot(list(range(len(iters[0]))), iters[0], "r-", label="p1")
plt.plot(list(range(len(iters[1]))), iters[1], "g-", label="p2")
plt.plot(list(range(len(iters[2]))), iters[2], "b-", label="p3")
plt.xlabel("percentage of noise")
plt.ylabel("n epochs")
plt.title(f'Average of number of iterations needed to converge')
plt.legend()
plt.savefig(fname=f'{SAVE_FOLDER}noise_distortion_02.eps', format='eps', bbox_inches='tight')
plt.savefig(fname=f'{SAVE_FOLDER}noise_distortion_02', bbox_inches='tight')
plt.close()
