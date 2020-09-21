import datetime
import math
import os
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np

from demo.util import prepare_dataset
from model.mlp import MLP
from utils.util import ensure_dir, mae

if __name__ == '__main__':
    ########################################
    #### MLP - sin(2x) and square(2x)  #####
    ########################################
    save_folder = "../../imgs/mlp/sinewave"
    ETA_VALUES = [0.002, 0.05, 0.1, 0.25, 1]
    NODES = [32]
    LOOPS = 4
    datasets = [
        # ["sine", prepare_dataset(lambda x: math.sin(2 * x))]
        # ,
        ["sine_noise", prepare_dataset(lambda x: math.sin(2 * x) + random.gauss(0, 0.1 ** 0.5))]
        # ,
        # ["square", prepare_dataset(lambda x: 1 if math.sin(2 * x) >= 0 else -1)]
        ,
        ["square_noise", prepare_dataset(lambda x: (1 if math.sin(2 * x) >= 0 else -1) + random.gauss(0, 0.1 ** 0.5))]
    ]

    results = {}
    for dataset_name, (train, valid, test) in datasets:
        batch_size = len(train[0])
        results[dataset_name] = {}
        for eta in ETA_VALUES:
            results[dataset_name][eta] = {}
            run_save_prefix = os.path.join(save_folder, f"{dataset_name}/e={eta}/")
            print(run_save_prefix)
            ensure_dir(run_save_prefix)

            for number_of_nodes in NODES:
                run_name = f'NODE{number_of_nodes:02d}_e={eta}_d={dataset_name}'.replace(".", ",")
                mae_array = []
                for i in range(LOOPS):
                    i += 1
                    print(run_name)
                    net = MLP(train[0], train[1], number_of_nodes, momentum=0, outtype="linear")
                    train_losses, valid_losses, _, _, pocket_epoch = net.train(
                        train[0], train[1], valid[0], valid[1], eta, 300000,
                        early_stop_count=100000, shuffle=False, batch_size=batch_size
                    )
                    net.forward(test[0])
                    print(mae(net.outputs, test[1]))
                    mae_array.append(mae(net.outputs, test[1]))
                mean, stddev = statistics.mean(mae_array), statistics.stdev(mae_array)

                results[dataset_name][eta][number_of_nodes] = mean, stddev

            plt.title(f'Performance e={eta} d={dataset_name}')
            plt.plot(NODES, np.log([results[dataset_name][eta][node][0] for node in NODES]))
            plt.xlabel('number of nodes')
            plt.ylabel('log of test MAE')
            plt.savefig(fname=os.path.join(f'{run_save_prefix}', f'node_performance'), dpi=300)
            plt.close()

    results_filename = f"results etas:{ETA_VALUES}" \
                       f" n:{NODES}" \
                       f" time:{datetime.datetime.now()}" \
                       f".txt"
    with open(os.path.join(save_folder, results_filename), "w") as f:
        print(os.path.join(save_folder, "results " + str(datetime.datetime.now()) + ".txt"))
        f.write(f"{results_filename}\n{results}")
    print(results)

    for dataset in results.keys():
        with open(os.path.join(save_folder, f"{dataset}_{datetime.datetime.now()}.csv"), 'a') as csv:
            for metric in [(0, "mean"), (1, "std dev")]:
                csv.write(f"{metric[1]}\n")
                csv.write(f"n,{','.join([str(x) for x in ETA_VALUES])}\n")
                for n in NODES:
                    csv.write(f"{n}")
                    for eta in ETA_VALUES:
                        csv.write(f",{results[dataset][eta][n][metric[0]]}")
                    csv.write("\n")
                csv.write("\n\n")
