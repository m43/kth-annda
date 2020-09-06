import numpy as np

from demo.perceptron.demo_util import delta_rule_learning_demo, \
    print_results_as_table, perpare_reproducable_inseparable_dataset_1
from utils.util import ensure_dir

if __name__ == '__main__':
    ##############################################
    #### DELTA. RULE for inseparable dataset #####
    ##############################################

    save_folder = "../../imgs"
    eta_values = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.1, 0.2, 0.3]
    max_iter = 10000  # max number of epochs
    debug = False
    plots = False
    eval = True
    delta_n = 50  # number of epochs without improvements in delta learning
    delta_n_batch = 150
    bias = True
    loops = 10
    inputs, targets = perpare_reproducable_inseparable_dataset_1()

    ensure_dir(save_folder)
    delta_results = {}
    for batch_size in [targets.shape[1]]:
        delta_results[batch_size] = {}
        for eta in eta_values:
            delta_results[batch_size][eta] = {}
            print(f"ETA is {eta}")

            accuracies = []
            losses = []
            convergence_epochs = []
            for i in range(loops):
                acc, loss, cepoch = delta_rule_learning_demo(
                    inputs, targets,
                    f"DELTA_RULE_INSEPARABLE_d:0_{'' if bias else '_NO_BIAS'}_b:{batch_size}_eta:{eta}_max_iter:{max_iter}".replace(
                        ".", ","), debug and i == 0, save_folder, max_iter, eta,
                    (delta_n if batch_size == 1 else delta_n_batch), batch_size, bias, plots_with_debug=plots,
                    confusion_matrix=eval and i == 0)
                cepoch += 1  # cepoch=0 means that it was the end of the first epoch, but we want this to be noted as 1
                accuracies.append(acc)
                losses.append(loss)
                convergence_epochs.append(cepoch)
                print(".", end="")

            m_acc, std_acc = np.array(accuracies).mean(), np.array(accuracies).std()
            m_loss, std_loss = np.array(losses).mean(), np.array(losses).std()
            m_cepoch, std_cepoch = np.array(convergence_epochs).mean(), np.array(convergence_epochs).std()

            delta_results[batch_size][eta]["accuracy"] = (m_acc, std_acc)
            delta_results[batch_size][eta]["loss"] = (m_loss, std_loss)
            delta_results[batch_size][eta]["co_epoch"] = (m_cepoch, std_cepoch)

            print("accuracies", accuracies)
            print("losses", losses)
            print("convergence_epochs", convergence_epochs)
            print(f"acc_mean={m_acc} acc_std={std_acc}")
            print(f"loss_mean={m_loss} loss_std={std_loss}")
            print(f"c_epoch_mean={m_cepoch} c_epoch_std={std_cepoch}")
            print()
        print()

    print("DELTA RULE")
    print("batch_size > eta > results")
    print(delta_results)

    for k, v in delta_results.items():
        print(f"B:{k}")
        print_results_as_table(delta_results[k], ["accuracy", "co_epoch", "loss"])
