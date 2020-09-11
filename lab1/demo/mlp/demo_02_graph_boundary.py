import os

import matplotlib.pyplot as plt
import numpy as np

from demo.util import two_class_conf_mat_metrics, \
    perpare_reproducable_inseparable_dataset_2_with_subsets
from model.mlp import MLP
from utils.util import ensure_dir, graph_surface, scatter_plot_2d_features

if __name__ == '__main__':
    #########################################
    #### Multi graph decision boundary  #####
    #########################################

    save_folder = "../../imgs"
    # eta_values = [0.002, 0.005, 0.01, 0.05, 0.1, 0.25]
    eta = 0.25
    epochs = 100000  # max number of epochs
    early_stop = 150
    shuffle = False
    loops = 4

    dataset, train_sets, valid_sets = perpare_reproducable_inseparable_dataset_2_with_subsets()
    ensure_dir(save_folder)

    t = dataset
    v = dataset

    train = (t[0].T.copy(), t[1].T.copy())
    valid = (v[0].T.copy(), v[1].T.copy())
    train[1][train[1] == -1] = 0  # cause of sigmoid output
    valid[1][valid[1] == -1] = 0
    batch_size = train[0].shape[0]

    # for n_hidden_nodes in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for n_hidden_nodes in [1, 2, 3, 4, 5, 8, 10, 25, 100, 1000]:
        name = f"MLP_2_b:{batch_size}_h:{n_hidden_nodes}_eta:{eta}_".replace(".", ",")
        print(name)
        net = MLP(train[0], train[1], n_hidden_nodes, momentum=0)
        train_losses, valid_losses, train_accuracies, valid_accuracies, pocket_epoch = net.train(
            train[0], train[1], valid[0], valid[1], eta, epochs,
            early_stop_count=early_stop, shuffle=shuffle, batch_size=batch_size
        )

        metrics = [
            "pocket_epoch",
            "train_loss", "train_acc", "train_sens", "train_spec",
            "valid_loss", "valid_acc", "valid_sens", "valid_spec"
        ]
        cm_train = net.confmat(train[0], train[1])
        cm_valid = net.confmat(valid[0], valid[1])
        train_acc, train_sens, train_spec, _, _ = two_class_conf_mat_metrics(cm_train)
        valid_acc, valid_sens, valid_spec, _, _ = two_class_conf_mat_metrics(cm_valid)
        train_loss, valid_loss = train_losses[pocket_epoch], valid_losses[pocket_epoch]

        for m in metrics:
            print(f"\t{m}:{locals()[m]}")
        # print(train_losses, valid_losses, train_accuracies, valid_accuracies, pocket_epoch)
        print(cm_train)
        print(cm_valid)
        print()


        # graph_surface(lambda x: x[:, 0] ** 2 + x[:, 1] - 10, ([-10, -10], [10, 10]), 0, 512, 512)
        # plt.show()

        def f(x):
            net.forward(x)
            o = np.where(net.outputs > 0.5, 1, 0)
            return o
        # def scatter_plot_2d_features(inputs, targets, name, line_coefficients=None, save_folder=None, show_plot=True):
        scatter_plot_2d_features(dataset[0], dataset[1], name, show_plot=False)
        graph_surface(f, ([-4, -4], [4, 4]), 0, 512, 512)
        plt.text(0, 0.1, f"t_acc:{train_acc}", ha='center')
        plt.text(0, 0.2, f"v_acc:{valid_acc}", ha='center')
        plt.text(0, 0.3, f"t_loss{train_loss}", ha='center')
        plt.text(0, 0.4, f"v_loss{valid_loss}", ha='center')
        plt.savefig(os.path.join(save_folder, name) + ".png", dpi=300)
        plt.show()
