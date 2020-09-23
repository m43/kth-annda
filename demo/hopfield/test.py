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





for img, current_step in zip(pics[0:3], range(3)):
    image = [img[i:i + int(PICTURE_SIZE ** (1 / 2))] for i in
             range(0, PICTURE_SIZE, int(PICTURE_SIZE ** (1 / 2)))]
    plt.imshow(image)
    plt.title(f'attractor={current_step}')
    plt.savefig(fname=f'{SAVE_FOLDER}attractor={current_step}')
    plt.close()


