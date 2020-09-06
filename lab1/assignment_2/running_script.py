import os
import time

node_argument = [2]
regularization_argument = [0]
neuron_argument = 0
layer_argument = 0
for i in node_argument:
    for j in regularization_argument:
        command = 'python main.py ' + str(i) + ' ' + str(j)
        print('Running:\n', command)
        os.system(command)
