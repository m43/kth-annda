import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from matplotlib import cm

from demo.util import prepare_dataset
from model.kmeans import KMeansEuclidean
from model.rbf import RBF
from utils.util import ensure_dir


def plot_function(test, outputs, run_save_prefix, run_name, centres_x=None, centres_y=None, sigmas=None, clusters=None):
    plt.title(f'{run_name}')
    plt.plot(test[0], test[1], label='true test values')
    plt.plot(test[0], outputs, label='prediction test values')
    if clusters is not None:
        plt.scatter(test[0], test[1], c=clusters, cmap=cm.rainbow, linewidth=3)
    if sigmas is not None:
        ax = plt.gca()
        for cx, cy, cs in zip(centres_x, centres_y, sigmas):
            circle = plt.Circle((cx, cy), cs, color='black', alpha=0.6, clip_on=False)
            ax.add_artist(circle)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(fname=os.path.join(f'{run_save_prefix}', f'{run_name}_test'), dpi=300)
    plt.close()


if __name__ == '__main__':
    ########################################
    #### RBF - sin(2x) and square(2x)  #####
    ########################################
    save_folder = "../../imgs/rbf/results_14_kmeans_no_magic_but_norm"
    nodes = [18, 25, 32]
    # nodes = list(range(1, 33))
    # nodes = [5, 10, 15, 20, 25]
    # 0 uses default variance for the uniform RBF initialization (a square root of the distance between RBF centers)
    # variances = [0]
    # variances = [0.1, 0.25, 0.5, 0.75, 1, 2]
    # variances = [0.5, 0.75]
    variances = [0.25]
    # eta_values = [0, 0.25]
    eta_values = [0.25]
    # eta_values = [0, 0.002, 0.05, 0.1, 0.25, 1]
    # eta_values = [0, 0.002, 0.05, 0.1]
    LOOPS = 4
    kmeans = True
    kmeans_normalized = False
    # kmeans_variances = [(0.1, 3), (0.2, 2), (0.3, 2), (0.3, 2.5)]
    kmeans_variances = [(0.2, 2)]

    datasets = [
        # ["sine", prepare_dataset(lambda x: math.sin(2 * x))]
        # ,
        ["sine_noise", prepare_dataset(lambda x: math.sin(2 * x) + random.gauss(0, 0.1 ** 0.5))],
        # ["square", prepare_dataset(lambda x: 1 if math.sin(2 * x) >= 0 else -1)],
        # ["square_noise", prepare_dataset(lambda x: (1 if math.sin(2 * x) >= 0 else -1) + random.gauss(0, 0.1 ** 0.5))]
    ]

    results = {}
    for dataset_name, (train, valid, test) in datasets:
        results[dataset_name] = {}
        for variance in variances if not kmeans else kmeans_variances:
            results[dataset_name][variance] = {}
            for eta in eta_values:
                results[dataset_name][variance][eta] = {}
                run_save_prefix = os.path.join(save_folder, f"{dataset_name}/v={variance}/e={eta}/")
                print(run_save_prefix)
                ensure_dir(run_save_prefix)

                for number_of_nodes in nodes:
                    run_name = f'NODE{number_of_nodes:02d}_v={variance}_e={eta}_d={dataset_name}'.replace(".", ",")

                    rbf = None
                    if kmeans:
                        mae = []
                        for _ in range(LOOPS):
                            kmeans = KMeansEuclidean(number_of_nodes, train[0].shape[1], one_winner=True)
                            rbf = RBF(number_of_nodes, train[0].shape[1], train[1].shape[1], rbf_init="kmeans",
                                      normalize_hidden_outputs=kmeans_normalized,
                                      rbf_init_data={"inputs": train[0], "kmeans": kmeans, "eta": 0.1, "n_iter": 2000,
                                                     "variance_rescale": variance})
                            valid_mae, pocket_epoch = rbf.delta_learning(train[0], train[1], eta, valid[0], valid[1])
                            print(f"N:{number_of_nodes} V:{variance} ETA:{eta} MAE:{valid_mae} P_EPOCH:{pocket_epoch}")
                            test_mae = rbf.evaluate_mae(test[0], test[1])
                            mae.append(test_mae)
                        mean, stddev = np.mean(mae), np.std(mae)
                    elif eta == 0:
                        data = np.concatenate((train[0], valid[0])), np.concatenate((train[1], valid[1]))
                        rbf = RBF(number_of_nodes, 1, 1, rbf_init="uniform", rbf_init_data=[0, 2 * math.pi, variance])
                        rbf.least_squares_training(data[0], data[1])
                        mean, stddev = rbf.evaluate_mae(test[0], test[1]), 0
                    else:
                        mae = []
                        for _ in range(LOOPS):
                            rbf = RBF(number_of_nodes, 1, 1, rbf_init="uniform", rbf_init_data=[0, 2 * np.pi, variance])
                            pocket_mae, pocket_epoch = rbf.delta_learning(train[0], train[1], eta, valid[0], valid[1])
                            print(f"N:{number_of_nodes} V:{variance} ETA:{eta} MAE:{pocket_mae} P_EPOCH:{pocket_epoch}")
                            mae.append(rbf.evaluate_mae(test[0], test[1]))
                        mean, stddev = np.mean(mae), np.std(mae)

                    centres_x, centres_y = rbf.rbf_weights[0], rbf.forward_pass(rbf.rbf_weights.T).T[0]
                    sigmas = rbf.rbf_variances ** 0.5
                    plot_function(test, rbf.forward_pass(test[0]), run_save_prefix, run_name,
                                  centres_x, centres_y, sigmas)
                    results[dataset_name][variance][eta][number_of_nodes] = mean, stddev

                plt.title(f'Performance v={variance} e={eta} d={dataset_name}')
                plt.plot(nodes, np.log10([results[dataset_name][variance][eta][node][0] for node in nodes]))
                plt.xlabel('number of nodes')
                plt.ylabel('log of test MAE')
                plt.savefig(fname=os.path.join(f'{run_save_prefix}', f'node_performance'), dpi=300)
                plt.close()

    results_filename = f"results etas:{eta_values}" \
                       f" n:{nodes}" \
                       f" time:{datetime.datetime.now()}" \
                       f".txt"
    with open(os.path.join(save_folder, results_filename), "w") as f:
        print(os.path.join(save_folder, "results " + str(datetime.datetime.now()) + ".txt"))
        f.write(f"{results_filename}\n{results}")
    print(results)

    # results[dataset_name][variance][eta][number_of_nodes] = mean, stddev
    for dataset in results.keys():
        with open(os.path.join(save_folder, f"{dataset}_{datetime.datetime.now()}.csv"), 'a') as csv:
            for metric in [(0, "mean"), (1, "std dev")]:
                csv.write(f"{metric[1]}\n")
                for var in variances if not kmeans else kmeans_variances:
                    csv.write(f"{var}\n")
                    csv.write(f"n,{','.join([str(x) for x in eta_values])}\n")
                    for n in nodes:
                        csv.write(f"{n}")
                        for eta in eta_values:
                            csv.write(f",{results[dataset][var][eta][n][metric[0]]}")
                        csv.write("\n")
                    csv.write("\n\n")
                csv.write("\n\n")
