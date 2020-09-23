import os

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../model/')
from hopfield import Hopfield
# from model.hopfield import Hopfield

PICTURE_SIZE = 1024
SAVE_FOLDER = 'demo_02/pictures/'
BATCH = True
DEBUG = False

N_SAMPLE_ATTRACTORS = 10


if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

print("Set weights at random.")
# create model
my_model = Hopfield(PICTURE_SIZE, debug_mode=DEBUG)
my_model.weights = np.random.randn(PICTURE_SIZE, PICTURE_SIZE)


gen_pattern = lambda: np.where(np.random.normal(size=1024) >= 0, 1, -1)
attractors = set()


# Try to find attractors
print("10 Random patterns sample to find attractors.")
for i in range(N_SAMPLE_ATTRACTORS):
    random_pattern = gen_pattern()
    my_model.set_state(random_pattern)
    state = my_model.update_automatically(batch=BATCH, update_cap=1000)
    print("x", end="")
    if state is not None:
        attractors.add(tuple(state))


print("Found ", len(attractors), " attractors, outputing images and printting energy")
for current_step in range(len(attractors)):
    state = list(attractors)[current_step]
    # print(state)
    energy = my_model.energy(state)
    print("\t ", current_step, " energy: ", energy)
    image = [state[i:i + int(PICTURE_SIZE ** (1 / 2))] for i in
             range(0, PICTURE_SIZE, int(PICTURE_SIZE ** (1 / 2)))]
    plt.imshow(image)
    plt.title(f'attractor={current_step}; energy={energy:.2f}')
    plt.savefig(fname=f'{SAVE_FOLDER}rnd_attractor={current_step}')
    plt.close()



print("Set weights at random but making the matrix symmetric")
my_model.weights = .5 * (my_model.weights + my_model.weights.T)
attractors = set()

# Try to find attractors
print("10 Random patterns sample to find attractors.")
for i in range(N_SAMPLE_ATTRACTORS):
    random_pattern = gen_pattern()
    my_model.set_state(random_pattern)
    state = my_model.update_automatically(batch=BATCH, update_cap=1000)
    print("x", end="")
    if state is not None:
        attractors.add(tuple(state))


print("Found ", len(attractors), " attractors, outputing images and printting energy")
for current_step in range(len(attractors)):
    state = list(attractors)[current_step]
    # print(state)
    energy = my_model.energy(state)
    print("\t ", current_step, " energy: ", energy)
    image = [state[i:i + int(PICTURE_SIZE ** (1 / 2))] for i in
             range(0, PICTURE_SIZE, int(PICTURE_SIZE ** (1 / 2)))]
    plt.imshow(image)
    plt.title(f'attractor={current_step}; energy={energy:.2f}')
    plt.savefig(fname=f'{SAVE_FOLDER}rnd_sym_attractor={current_step}')
    plt.close()