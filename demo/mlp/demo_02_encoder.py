import os

import matplotlib.pyplot as plt
import numpy as np

from model.mlp import MLP
from utils.util import ensure_dir

if __name__ == '__main__':
    #####################################
    #### MLP - The encoder problem  #####
    #####################################

    save_folder = "../../imgs/mlp/demo_03_6_not_sparse"
    ensure_dir(save_folder)
    hidden_nodes = [2]
    # eta_values = [0.001, 0.002, 0.004, 0.005, 0.01, 0.05, 0.1, 0.25, 1, 10]
    # eta_values = [0.002, 0.005, 0.01, 0.05, 0.1, 0.25, 1, 10]
    # eta_values = [0.25, 1, 5, 10, 25]
    eta_values = [0.25, 1, 5]
    momentum = 0
    epochs = 40000  # max number of epochs
    early_stopping_threshold = 1e-10
    early_stop = 200
    # batch_size choice is hardcoded in the loop
    shuffle = True
    debug_best_and_plot = True
    loops = 1

    n = 8
    np.random.seed(72090)
    x = np.where(np.random.random(64) > 0.5, 1, 0).reshape((8, 8))
    dataset = (x, x.copy())
    # dataset = (np.identity(n), np.identity(n))

    i = 1
    for n_hidden_nodes in hidden_nodes:
        print(f"x-----> {n_hidden_nodes} hidden nodes <-----x")
        for eta in eta_values:
            for batch_size in [dataset[0].shape[0], 1]:
                for _ in range(loops):
                    name = f"MLP{i:05}_3_n-{n}_b-{batch_size}_h-{n_hidden_nodes}_eta-{eta}_m-{momentum}".replace(".",
                                                                                                                 ",")
                    i += 1

                    print(f"ETA is {eta}")
                    print(name)

                    net = MLP(dataset[0], dataset[1], n_hidden_nodes, momentum=momentum)
                    train_losses, _, train_accuracies, _, pocket_epoch = net.train(
                        dataset[0], dataset[1], dataset[0], dataset[1], eta, epochs,
                        early_stop_count=early_stop, shuffle=shuffle, batch_size=batch_size,
                        early_stopping_threshold=early_stopping_threshold
                    )
                    train_accuracies = np.array(train_accuracies) * 100 / n
                    net.forward(dataset[0])

                    print("pocket epoch", pocket_epoch)
                    print(f"w1:\n{net.weights1}\nw2:{net.weights2}\nnet.hidden:\n{net.hidden}\noutputs:{net.outputs}\n")

                    save_prefix = os.path.join(save_folder, name)
                    fig, ax1 = plt.subplots()
                    plt.title(name + " Learning curve")
                    x = np.arange(len(train_losses))
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Log MSE loss')
                    ax1.plot(x, np.log(train_losses), color='tab:red', label="Training loss", linewidth=2)
                    ax1.scatter(pocket_epoch, np.log(train_losses[pocket_epoch]), color="green")
                    ax1.tick_params(axis='y')
                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                    ax2.set_ylabel('Accuracy')
                    ax2.plot(x, train_accuracies, color="tab:blue", label="Training accuracy")
                    ax2.scatter(pocket_epoch, train_accuracies[pocket_epoch], color="green")
                    ax2.tick_params(axis='y')
                    fig.legend(bbox_to_anchor=(0.8, 0.7), loc="upper right")
                    textblock = f't_acc:{train_accuracies[pocket_epoch]:.2f}\n' \
                                f"t_loss:{train_losses[pocket_epoch]:.4e}\n" \
                                f"pocket_epoch:{pocket_epoch}"
                    plt.text(0.01, 0.02, textblock, fontdict={'family': 'monospace'}, transform=ax1.transAxes)
                    plt.savefig(f"{save_prefix}_learning_curve.png", dpi=300)
                    plt.show()

"""
8-3-8
net.weights1:
 -3.1   5.3  -4.9 
  1.5  -5.2  -5.0 
  6.4   6.2   5.5 
 -3.7  -4.6   4.9 
 -5.8   4.4   4.9 
  5.0  -5.9   3.5 
  4.9   2.5  -6.1 
 -5.9  -2.3  -3.2 
 -0.2  -0.3  -0.0
net.hidden:
    0 1 0
    1 0 0
    1 1 1
    0 0 1
    0 1 1
    1 0 1
    1 1 0
    0 0 0
net.weights2:
-11.2  12.4  10.6 -12.0 -12.0  11.9  12.8 -16.6 
 12.6 -16.2  10.6 -12.5  11.2 -12.1   9.4 -12.4 
-12.3 -14.9   9.7  11.7  11.0   9.0 -12.8 -12.7 
 -6.4  -4.0 -25.6  -5.6 -16.5 -15.0 -15.6   6.9 
 
8-2-8
ETA is 0.25
MLP00011_3_n:8_b:8_h:2_eta:0,25_m:0,9
pocket epoch 99999

net.weights1:
 -3.0  -1.2 
 -9.0   0.4 
  0.2   9.2 
  0.7  -3.2 
  5.0   1.6 
  8.9  -0.5 
 -1.0  -8.9 
 -1.7   2.2 
  0.3  -0.3
net.hidden:
  0.1   0.2
  0.0   0.5
  0.6   1.0
  0.7   0.0
  1.0   0.8
  1.0   0.3
  0.3   0.0
  0.2   0.9
net.weights2:
-55.2 -66.2   5.2  26.0  34.3  50.7 -16.0 -35.0 
-33.9  12.7  48.9 -55.1  19.2 -18.1 -69.6  46.4 
 13.7  -2.8 -48.0 -13.1 -44.9 -40.8   9.4 -29.5 
 
 
Printed with:
import re
from ast import literal_eval
import numpy as np

w1 = re.sub(r"([^[])\s+([^]])", r"\1, \2", w1)
w1 = np.array(literal_eval(w1))
net_hidden = re.sub(r"([^[])\s+([^]])", r"\1, \2", net_hidden)
net_hidden = np.array(literal_eval(net_hidden))
w2 = re.sub(r"([^[])\s+([^]])", r"\1, \2", w2)
w2 = np.array(literal_eval(w2))
outputs = re.sub(r"([^[])\s+([^]])", r"\1, \2", outputs)
outputs = np.array(literal_eval(outputs))

for c in [w1, net_hidden, w2, outputs]:
    for a in c:
        for b in a:
            print(f"{b:5.1f}", end=" ")
        print()
"""
