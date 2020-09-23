import os

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../model/')
from hopfield import Hopfield
# from model.hopfield import Hopfield

PICTURE_SIZE = 1024
SAVE_FOLDER = 'demo_02/pictures/'
BATCH = False
DEBUG = False


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

# check stability of learned patterns
for i in range(3):
    print(f'\tChecking stability of picture p{i + 1}... ', end='')
    my_model.set_state(pics[i])
    new_state = my_model.update_step(batch=BATCH)
    if (new_state == pics[i]).all():
        print('Stable!')
    else:
        print('Not stable!!!')

# check pattern completion capabilities
degraded_pics = np.array([pics[9], pics[10]])  # (pictures 10 (degraded p1) and 11 (mix of p2 and p3))
print('\nChecking pattern completion capabilities')
for degraded_pic in degraded_pics:
    my_model.set_state(degraded_pic)
    new_state = my_model.update_automatically(batch=BATCH)
    if new_state is not None:
        print(f'\tFinished updating - ', end='')
        found_idx = np.where((pics[0:3] == new_state).all(axis=1))
        if len(found_idx[0]) != 0:
            print(f'stable state is equal to p{found_idx[0] + 1}')
        else:
            print('stable state does not belong to any learned pattern.')
    else:
        print('\tUpdating failed to converge!')


# define callback function for drawing pictures
def picture_callback(current_state, current_step):
    if current_step % 128 == 0:
        image = [current_state[i:i + int(PICTURE_SIZE ** (1 / 2))] for i in
                 range(0, PICTURE_SIZE, int(PICTURE_SIZE ** (1 / 2)))]
        plt.imshow(image)
        plt.title(f'Step={current_step}')
        plt.savefig(fname=f'{SAVE_FOLDER}pic_step={current_step:06d}')
        plt.close()


# check random pattern for results
random_pattern = np.random.normal(size=1024)
random_pattern = np.where(random_pattern >= 0, 1, -1)
print('\nUsing random picture to see convergence...')
my_model.set_state(random_pattern)
my_model.update_automatically(batch=BATCH, step_callback=picture_callback)
print('Done.')
