import os
import time

node_argument = [6, 8]
regularization_argument = [0.0, 0.01, 1.0]
noise_sigma = [0.09]

# run without noise
if not noise_sigma:
    for i in node_argument:
        for j in regularization_argument:
            command = 'python main.py ' + str(i) + ' ' + str(j)
            print('Running:\n', command)
            os.system(command)

# run with noise
else:
    for i in node_argument:
        for j in regularization_argument:
            for noise in noise_sigma:
                command = 'python main.py ' + str(i) + ' ' + str(j) + ' ' + str(noise)
                print('Running:\n', command)
                os.system(command)
