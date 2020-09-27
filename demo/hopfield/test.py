import matplotlib.pyplot as plt
import numpy as np

from utils.util import ensure_dir

PICTURE_SIZE = 1024
SAVE_FOLDER = '../../imgs/hopfield/test/'
BATCH = False
DEBUG = False

ensure_dir(SAVE_FOLDER)

# load pictures into a numpy float 2D array, 1 row per picture
with open('../../datasets/pict.dat', 'r') as source:
    data = source.read().strip('\n')
    data = data.split(',')
    pics = np.array([np.array(data[i:i + PICTURE_SIZE]).astype(np.float) for i in range(0, len(data), 1024)])
print(f'Successfully loaded {len(pics)} pictures.\n')

for img, current_step in zip(pics[0:11], range(11)):
    image = [img[i:i + int(PICTURE_SIZE ** (1 / 2))] for i in
             range(0, PICTURE_SIZE, int(PICTURE_SIZE ** (1 / 2)))]
    plt.imshow(image)
    plt.title(f'attractor={current_step}')
    plt.savefig(fname=f'{SAVE_FOLDER}attractor={current_step}')
    plt.close()
