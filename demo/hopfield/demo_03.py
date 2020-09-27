import matplotlib.pyplot as plt
import numpy as np

from model.hopfield import Hopfield
from utils.util import ensure_dir

PICTURE_SIZE = 1024
SAVE_FOLDER = '../../imgs/hopfield/demo_03/'
BATCH = True
DEBUG = False

N_SAMPLE_ATTRACTORS = 1000

ensure_dir(SAVE_FOLDER)

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
# my_model.weights *= 2 # Note: doubling the weights doubles the energy values

gen_pattern = lambda: np.where(np.random.normal(size=1024) >= 0, 1, -1)
# attractors = {tuple(i) for i in pics[0:3]}


# Try to find attractors
print(f"{N_SAMPLE_ATTRACTORS} Random patterns sample to find new attractors.")
for i in range(N_SAMPLE_ATTRACTORS):
    random_pattern = gen_pattern()
    my_model.set_state(random_pattern)
    state = my_model.update_automatically(batch=BATCH)
    if state is not None:
        attractors.add(tuple(state))


print("Found ", len(attractors), " attractors, outputing images and printing energy")
for current_step in range(len(attractors)):
    state = list(attractors)[current_step]
    energy = my_model.energy(state)
    print("\t ", current_step, " energy: ", energy)
    image = [state[i:i + int(PICTURE_SIZE ** (1 / 2))] for i in
             range(0, PICTURE_SIZE, int(PICTURE_SIZE ** (1 / 2)))]
    plt.imshow(image)
    plt.title(f'attractor={current_step}; energy={energy:.2f}')
    plt.savefig(fname=f'{SAVE_FOLDER}attractor={current_step}', bbox_inches='tight')
    plt.close()

print("\n")
print("Energy for distorted patterns")
print("\t p10:", my_model.energy(pics[9]))
print("\t p11:", my_model.energy(pics[10]))


energies = []
energy_list_callback = lambda x, _: energies.append(my_model.energy(x))

print("\n")
print("Using the sequential rule to approach an attractor and check the energy level")
random_pattern = gen_pattern()
print("starting energy: ", my_model.energy(random_pattern))
my_model.set_state(random_pattern)
my_model.update_automatically(batch=BATCH, step_callback=energy_list_callback)
plt.plot(list(range(len(energies))), energies)
plt.title(f'Convergence using sequential rule to approach an attractor')
plt.xlabel('update')
plt.ylabel('state energy')
plt.yscale('symlog')
plt.savefig(fname=f'{SAVE_FOLDER}convergence_01', bbox_inches='tight')
plt.close()
# plt.show()
print('Done.')

