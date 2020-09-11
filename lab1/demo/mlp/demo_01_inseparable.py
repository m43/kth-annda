import os

import matplotlib.pyplot as plt
import numpy as np

from demo.util import print_results_as_table, two_class_conf_mat_metrics, \
    perpare_reproducable_inseparable_dataset_2_with_subsets
from model.mlp import MLP
from utils.util import ensure_dir, scatter_plot_2d_features, graph_surface


def show_best(best, save_folder, train, valid):
    def f(x):
        best['net'].forward(x)
        return np.where(best['net'].outputs > 0.5, 1, 0)

    print(f"Best: v_acc:{best['best_acc']} v_loss:{best['best_loss']} name:{best['name']}\n{best}")
    save_prefix = os.path.join(save_folder, best['name'])
    fig, ax1 = plt.subplots()
    plt.title(best["name"] + " Learning curve")
    x = np.arange(len(best["tl"]))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Log MSE loss')
    ax1.plot(x, np.log(best["tl"]), color='tab:red', label="Training loss")
    ax1.plot(x, np.log(best["vl"]), color='tab:orange', label="Validation loss")
    ax1.scatter(best["pocket_epoch"], np.log(best["best_loss"]), color="green")
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Accuracy')
    ax2.plot(x, best["ta"], color="tab:blue", label="Training accuracy")
    ax2.plot(x, best["va"], color="tab:cyan", label="Validation accuracy")
    ax2.scatter(best["pocket_epoch"], best["best_acc"], color="green")
    ax2.tick_params(axis='y')
    fig.legend(bbox_to_anchor=(0.8, 0.7), loc="upper right")
    plt.savefig(f"{save_prefix}_learning_curve.png", dpi=300)
    plt.show()
    scatter_plot_2d_features(train[0].T, train[1].T, best['name'] + " Decision boundary",
                             show_plot=False,
                             fmt=("bo", "ro"))
    scatter_plot_2d_features(valid[0].T, valid[1].T, best['name'] + " Decision boundary",
                             show_plot=False,
                             fmt=("bx", "rx"))
    graph_surface(f, ([-5, -5], [5, 5]), 0, 512, 512)
    textblock = f"t_acc:{best['train_acc'] * 100:.2f}\n" \
                f"v_acc:{best['valid_acc'] * 100:.2f}\n" \
                f"t_loss:{best['train_loss']:.4e}\n" \
                f"v_loss:{best['valid_loss']:.4e}\n" \
                f"pocket_epoch:{best['pocket_epoch']}"
    plt.text(0.01, 0.02, textblock, fontdict={'family': 'monospace'}, transform=plt.gca().transAxes)
    plt.savefig(f"{save_prefix}_decision_boundary.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    #############################################
    #### MLP on inseparable two class data  #####
    #############################################

    save_folder = "../../imgs/mlp/demo_01_6_no_early_stopping_for_d:3"
    ensure_dir(save_folder)
    hidden_nodes = [3]
    # hidden_nodes = [5, 8, 10, 100]
    # hidden_nodes = [100, 150, 200, 250]
    # eta_values = [0.001, 0.002, 0.004, 0.005, 0.01, 0.05, 0.1, 0.25, 1, 10]
    # eta_values = [0.002, 0.005, 0.01, 0.05, 0.1, 0.25, 1, 10]
    # eta_values = [0.002, 0.05, 0.1, 0.25, 1]
    eta_values = [0.05]
    epochs = 100000  # max number of epochs
    early_stop = 400000000
    batch = True
    shuffle = False
    loops = 1
    debug_best_and_plot = True

    dataset, train_sets, valid_sets = perpare_reproducable_inseparable_dataset_2_with_subsets()
    train_sets.append(dataset)  # To have the original dataset with train=valid (aka no validation)
    valid_sets.append(dataset)

    i = 1
    for dataset_idx in [3]:
        print(f"{'#' * 60}\n{'#' * 29}{dataset_idx:^3}{'#' * 28}\n{'#' * 60}")

        train = (train_sets[dataset_idx][0].T.copy(), train_sets[dataset_idx][1].T.copy())
        valid = (valid_sets[dataset_idx][0].T.copy(), valid_sets[dataset_idx][1].T.copy())
        train[1][train[1] == -1] = 0  # cause of sigmoid output
        valid[1][valid[1] == -1] = 0

        batch_size = train[0].shape[0] if batch else 1
        results = {}
        metrics = [
            "pocket_epoch",
            "train_loss", "train_acc", "train_sens", "train_spec",
            "valid_loss", "valid_acc", "valid_sens", "valid_spec"
        ]
        for n_hidden_nodes in hidden_nodes:
            print(f"x-----> {n_hidden_nodes} hidden nodes <-----x")
            results[n_hidden_nodes] = {}
            # best = {"best_acc": 0, "best_loss": np.inf}
            for eta in eta_values:
                print(f"ETA is {eta}")
                results[n_hidden_nodes][eta] = {}
                accumulated_metrics = {}
                for m in metrics:
                    accumulated_metrics[m] = []

                name = f"MLP{i:05}_d:{dataset_idx}_b:{batch_size}_h:{n_hidden_nodes}_eta:{eta}".replace(".", ",")
                i+=1
                print(name)
                best = {"best_acc": 0, "best_loss": np.inf}
                for _ in range(loops):
                    print(".", end="")

                    net = MLP(train[0], train[1], n_hidden_nodes, momentum=0)
                    train_losses, valid_losses, train_accuracies, valid_accuracies, pocket_epoch = net.train(
                        train[0], train[1], valid[0], valid[1], eta, epochs,
                        early_stop_count=early_stop, shuffle=shuffle, batch_size=batch_size
                    )
                    cm_train = net.confmat(train[0], train[1])
                    cm_valid = net.confmat(valid[0], valid[1])
                    print("cm_train", cm_train)
                    print("cm_valid", cm_valid)
                    print("pocket epoch", pocket_epoch)

                    train_loss, valid_loss = train_losses[pocket_epoch], valid_losses[pocket_epoch]
                    train_acc, train_sens, train_spec, _, _ = two_class_conf_mat_metrics(
                        net.confmat(train[0], train[1]))
                    valid_acc, valid_sens, valid_spec, _, _ = two_class_conf_mat_metrics(
                        net.confmat(valid[0], valid[1]))

                    if debug_best_and_plot:
                        if valid_acc > best["best_acc"] or (
                                valid_acc == best["best_acc"] and valid_loss < best["best_loss"]):
                            print("BEST!")
                            best.update(
                                {"name": name, "best_acc": valid_acc, "best_loss": valid_loss, "net": net,
                                 "tl": train_losses, "vl": valid_losses, "ta": train_accuracies, "va": valid_accuracies,
                                 "tcm": cm_train, "vcm": cm_valid})
                            for m in metrics:
                                best[m] = locals()[m]

                    for m in metrics:
                        accumulated_metrics[m].append(locals()[m])
                print()
                for m in metrics:
                    results[n_hidden_nodes][eta][m] = np.array(accumulated_metrics[m]).mean(), np.array(
                        accumulated_metrics[m]).std()

                if debug_best_and_plot:
                    show_best(best, save_folder, train, valid)

            print(results)
            print(".\n.\n.")

        print(f"Dataset:{dataset_idx} (+1)")
        for k, v in results.items():
            print(f"n_hidden:{k}")
            print_results_as_table(results[k], metrics)
