import matplotlib.pyplot as plt
import numpy as np

from model.hopfield import Hopfield
from utils.util import ensure_dir

PICTURE_SIZE = 1024
SAVE_FOLDER = '../../imgs/hopfield/demo_02_4patterns_2/'
BATCH = True
DEBUG = True

ensure_dir(SAVE_FOLDER)

# load pictures into a numpy float 2D array, 1 row per picture
with open('../../datasets/pict.dat', 'r') as source:
    data = source.read().strip('\n')
    data = data.split(',')
    pics = np.array([np.array(data[i:i + PICTURE_SIZE]).astype(np.float) for i in range(0, len(data), 1024)])
print(f'Successfully loaded {len(pics)} pictures.\n')

# Define the patterns to be learned
patterns_to_learn = pics[0:4]

# create model
my_model = Hopfield(PICTURE_SIZE, debug_mode=DEBUG)

# learn first three pictures
print(f"Learning pictures from p1 to p{len(patterns_to_learn) + 1}.")
my_model.learn_patterns(patterns_to_learn)

# check stability of learned patterns
for i in range(len(patterns_to_learn)):
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
        found_idx = np.where((patterns_to_learn == new_state).all(axis=1))
        if len(found_idx[0]) != 0:
            print(f'stable state is equal to p{found_idx[0] + 1}')
        else:
            print('stable state does not belong to any learned pattern.')
    else:
        print('\tUpdating failed to converge!')


# define callback function for drawing pictures
def picture_callback(current_state, current_step, name=""):
    if current_step % 128 == 0:
        image = [current_state[i:i + int(PICTURE_SIZE ** (1 / 2))] for i in
                 range(0, PICTURE_SIZE, int(PICTURE_SIZE ** (1 / 2)))]
        plt.imshow(image)
        plt.title(f'Step={current_step}')
        plt.savefig(fname=f'{SAVE_FOLDER}{name}pic_step={current_step:06d}', bbox_inches='tight')
        plt.close()


for i, pattern in enumerate(patterns_to_learn):
    my_model.set_state(pattern)
    my_model.update_automatically(batch=BATCH, step_callback=lambda a, b: picture_callback(a, b, f"pattern{i}___"))

# check random pattern for results
random_pattern = np.random.normal(size=1024)
random_pattern = np.where(random_pattern >= 0, 1, -1)
print('\nUsing random picture to see convergence...')
my_model.set_state(random_pattern)
my_model.update_automatically(batch=BATCH, step_callback=picture_callback)
print('Done.')
