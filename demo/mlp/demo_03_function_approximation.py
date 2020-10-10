import datetime
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from model.mlp import MLP
from utils.util import ensure_dir

"""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
"""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

save_folder = "../../imgs/mlp/demo_03"
ensure_dir(save_folder)


def demo_run(t):
    ########################################
    #### MLP - Function approximation  #####
    ########################################

    n_hidden = t
    split = 0.6

    # hidden_nodes = [1, 2, 3, 5, 8, 10, 15, 20, 22, 25, 100]
    # hidden_nodes = list(range(1, 25 + 1)) + [100]
    hidden_nodes = [n_hidden]
    # eta_values = [0.002, 0.005, 0.01, 0.05, 0.1, 0.25, 1, 10]
    eta_values = [0.25, 0.5]
    epochs = 1000  # max number of epochs
    early_stop = 400000000  # no early stopping
    batch = True
    shuffle = False
    loops = 4
    debug = True
    momentum = 0
    show_plots = False

    step = 0.25
    np.random.seed(72)
    x = np.arange(-5, 5, step)
    y = np.arange(-5, 5, step)
    x, y = np.meshgrid(x, y)
    z = np.exp(-(x ** 2 + y ** 2) / 10) - 0.5
    xy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    dataset = xy.copy(), z.reshape(-1, 1).copy()

    n = dataset[0].shape[0]
    shuffler = np.random.permutation(n)
    shuffled_dataset = dataset[0][shuffler], dataset[1][shuffler]
    train = shuffled_dataset[0][0:int(n * split)], shuffled_dataset[1][0:int(n * split)]
    # valid = shuffled_dataset[0][int(n * split):], shuffled_dataset[1][int(n * split):]
    valid = shuffled_dataset

    metrics = ["pocket_epoch", "train_loss", "valid_loss"]

    i = 100
    batch_size = train[0].shape[0] if batch else 1
    results = {}  # results[n_hidden][eta][metric]
    for n_hidden_nodes in hidden_nodes:
        print(f"x-----> {n_hidden_nodes} hidden nodes <-----x")
        results[n_hidden_nodes] = {}
        for eta in reversed(eta_values):
            print(f"ETA is {eta}")
            results[n_hidden_nodes][eta] = {}
            accumulated_metrics = {}
            for m in metrics:
                accumulated_metrics[m] = []
            print(f"Loop starting. {loops} loops left to go.")
            for loop_idx in range(loops):
                name = f"MLP{i:05}_GAUSS" \
                       f"_h-{n_hidden_nodes}" \
                       f"_eta-{eta}" \
                       f"_b-{batch_size}{'' if momentum == 0 else 'momentum:' + str(momentum)}".replace(".", ",")
                i += 1
                print(name)
                net = MLP(train[0], train[1], n_hidden_nodes, momentum=momentum, outtype="linear")
                train_losses, valid_losses, _, _, pocket_epoch = net.train(
                    train[0], train[1], valid[0], valid[1], eta, epochs,
                    early_stop_count=early_stop, shuffle=shuffle, batch_size=batch_size
                )
                train_loss, valid_loss = train_losses[pocket_epoch], valid_losses[pocket_epoch]
                print(f"pocket epoch: {pocket_epoch}\n"
                      f"train_loss:{train_loss}\n"
                      f"valid_loss:{valid_loss}")
                for m in metrics:
                    accumulated_metrics[m].append(locals()[m])
                if debug:

                    save_prefix = os.path.join(save_folder, str(n_hidden_nodes))
                    ensure_dir(save_prefix)
                    save_prefix = os.path.join(save_prefix, str(eta))
                    ensure_dir(save_prefix)
                    save_prefix = os.path.join(save_prefix, str(batch_size))
                    ensure_dir(save_prefix)
                    save_prefix = os.path.join(save_prefix, name)

                    fig, ax1 = plt.subplots()
                    plt.title(name + " Learning curve")
                    x_values = np.arange(len(train_losses))
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Log MSE loss')
                    ax1.plot(x_values, np.log(train_losses), color='tab:red', label="Training loss", linewidth=2)
                    ax1.plot(x_values, np.log(valid_losses), color='tab:orange', label="Validation loss")
                    ax1.scatter(pocket_epoch, np.log(valid_loss), color="green")
                    ax1.tick_params(axis='y')
                    fig.legend(bbox_to_anchor=(0.85, 0.7), loc="upper right")
                    plt.savefig(f"{save_prefix}_learning_curve.png", dpi=300)
                    if show_plots:
                        plt.show()
                    plt.close()

                    def f(inputs):
                        net.forward(inputs)
                        n = np.int(np.sqrt(inputs.shape[0]))
                        return net.outputs.reshape((n, n))

                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    surf = ax.plot_surface(x, y, f(xy), cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.8)

                    # Customize the z axis.
                    ax.set_zlim(-1.01, 1.01)
                    ax.zaxis.set_major_locator(LinearLocator(10))
                    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

                    # Add a color bar which maps values to colors.
                    fig.colorbar(surf, shrink=0.5, aspect=5)
                    plt.savefig(f"{save_prefix}_function_approx.png", dpi=300)
                    if show_plots:
                        plt.show()
                    plt.close()

            print("\n\n")
            for m in metrics:
                results[n_hidden_nodes][eta][m] = np.array(accumulated_metrics[m]).mean(), np.array(
                    accumulated_metrics[m]).std()

    results_filename = f"results m:{momentum}" \
                       f" etas:{eta_values}" \
                       f" nh:{hidden_nodes}" \
                       f" time:{datetime.datetime.now()}" \
                       f".txt"
    with open(os.path.join(save_folder, results_filename), "w") as f:
        print(os.path.join(save_folder, "results " + str(datetime.datetime.now()) + ".txt"))
        print(results)
        f.write(f"{results_filename}\n{results}")
    print(results)
    print(".\n.\n.")

    sep = ","
    for metric in metrics:
        for metric_idx in range(2):  # 0-->mean 1-->std
            # print table name
            print(f"Metric: {metric} {'mean' if metric_idx == 0 else 'std'}")

            # first row - table header
            print("eta", end="")
            for eta in eta_values:
                print(sep + str(eta), end="")
            print()

            # other rows - one for each number of hidden nodes
            for nh in hidden_nodes:
                print(nh, end="")
                for eta in eta_values:
                    print(sep + str(results[nh][eta][metric][metric_idx]), end="")
                print()
            print()
        print()
        print()


if __name__ == '__main__':
    print("go")
    with multiprocessing.Pool() as pool:
        ae = list(range(1, 11))
        print(pool.map(demo_run, ae))
