import os

import numpy as np

from demo.perceptron.demo_util import perpare_reproducable_inseparable_dataset_2, print_results_as_table, \
    two_class_conf_mat_metrics, sample_two_class_dataset
from model.delta_rule_perceptron import DeltaRulePerceptron
from utils.util import ensure_dir

if __name__ == '__main__':
    ##############################################
    #### DELTA. RULE for inseparable dataset #####
    ##############################################

    save_folder = "../../imgs"
    eta_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.1]
    max_iter = 10000  # max number of epochs
    delta_n = 50  # number of epochs without improvements in delta learning
    delta_n_batch = 150
    loops = 100
    inputs, targets = perpare_reproducable_inseparable_dataset_2()

    inputs_1, targets_1 = sample_two_class_dataset(inputs, targets, 25, 25)
    inputs_2, targets_2 = sample_two_class_dataset(inputs, targets, 50, 0)
    inputs_3, targets_3 = sample_two_class_dataset(inputs, targets, 0, 50)
    idx_4 = []
    p_a_below = 0.2
    p_a_above = 0.8
    for i, t in enumerate(targets[0]):
        if t == 1:
            if inputs[0, i] < 0 and np.random.random() < p_a_below:
                idx_4.append(i)
            if inputs[0, i] > 0 and np.random.random() < p_a_above:
                idx_4.append(i)
    inputs_4, targets_4 = np.delete(inputs, idx_4, axis=1), np.delete(targets, idx_4, axis=1)

    ensure_dir(save_folder)
    for dataset_idx, (current_inputs, current_targets) in enumerate(zip(
            [inputs, inputs_1, inputs_2, inputs_3, inputs_4], [targets, targets_1, targets_2, targets_3, targets_4])):
        print(f"{'#' * 60}\n{'#' * 29}{dataset_idx:^3}{'#' * 28}\n{'#' * 60}")
        delta_results = {}
        metrics = [
            "train_loss", "cepoch",
            "train_acc", "train_sens", "train_spec",
            "real_acc", "real_sens", "real_spec"
        ]
        for batch_size in [current_targets.shape[1]]:
            delta_results[batch_size] = {}
            for eta in eta_values:
                print(f"ETA is {eta}")
                delta_results[batch_size][eta] = {}

                acumulated_metrics = {}
                for m in metrics:
                    acumulated_metrics[m] = []

                name = f"_DELTA_RULE_INSEPARABLE_d:{dataset_idx}_BATCH_eta:{eta}_max_iter:{max_iter}".replace(".", ",")
                print(name)
                for i in range(loops):
                    print(".", end="")
                    perceptron = DeltaRulePerceptron(current_inputs, current_targets, False,
                                                     os.path.join(save_folder, name))
                    weights_per_epoch, train_acc, train_loss, cepoch = perceptron.train(current_inputs, current_targets, eta, max_iter,
                                                                                        batch_size,
                                                                                        shuffle=True,
                                                                                        stop_after=delta_n_batch)
                    cepoch += 1  # cepoch=0 means that it was the end of the first epoch, but we want this to be noted as 1
                    _, train_sens, train_spec, _, _ = two_class_conf_mat_metrics(
                        perceptron.conf_mat(current_inputs, current_targets))
                    real_acc, real_sens, real_spec, _, _ = two_class_conf_mat_metrics(
                        perceptron.conf_mat(inputs, targets))

                    for m in metrics:
                        acumulated_metrics[m].append(locals()[m])

                print()
                for m in metrics:
                    delta_results[batch_size][eta][m] = np.array(acumulated_metrics[m]).mean(), np.array(
                        acumulated_metrics[m]).std()

            print(".")
            print(".")
            print(".")

        for k, v in delta_results.items():
            print(f"B:{k}")
            print_results_as_table(delta_results[k], metrics)
