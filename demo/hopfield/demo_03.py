import os

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../model/')
from hopfield import Hopfield
# from model.hopfield import Hopfield

PICTURE_SIZE = 1024
SAVE_FOLDER = 'demo_03/pictures/'
BATCH = False
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



gen_pattern = lambda: np.where(np.random.normal(size=1024) >= 0, 1, -1)
attractors = {tuple(i) for i in pics[0:3]}

# Try to find attractors
print("1000 Random patterns sample to find new attractors.")
for i in range(N_SAMPLE_ATTRACTORS):
    random_pattern = gen_pattern()
    my_model.set_state(random_pattern)
    state = my_model.update_automatically(batch=BATCH)
    if state is not None:
        attractors.add(tuple(state))


print("Found ", len(attractors), " attractors, outputing images and printting energy")
for current_step in range(len(attractors)):
    state = list(attractors)[current_step]
    energy = my_model.energy(state)
    print("\t ", current_step, " energy: ", energy)
    image = [state[i:i + int(PICTURE_SIZE ** (1 / 2))] for i in
             range(0, PICTURE_SIZE, int(PICTURE_SIZE ** (1 / 2)))]
    plt.imshow(image)
    plt.title(f'new_attractor={current_step}; energy={energy:.2f}')
    plt.savefig(fname=f'{SAVE_FOLDER}attractor={current_step}')
    plt.close()


print("\n")
print("Energy for distorted patterns")
print("\t p10:", my_model.energy(pics[9]))
print("\t p11:", my_model.energy(pics[10]))


# def picture_callback(current_state, current_step):

# def print_energy_callback():

  
energies = []
energy_list_callback = lambda x,_: energies.append(my_model.energy(x))


print("\n")
print("Using the sequential rule to approach an attractor and check the energy level")
random_pattern = gen_pattern()
print("starting energy: ", my_model.energy(random_pattern))
my_model.set_state(random_pattern)
my_model.update_automatically(batch=BATCH, step_callback=energy_list_callback)

plt.plot(list(range(len(energies))), energies)
plt.show()
print('Done.')