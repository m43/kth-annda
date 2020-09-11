import numpy as np

from demo.util import print_results_as_table, two_class_conf_mat_metrics, \
    perpare_reproducable_inseparable_dataset_2_with_subsets
from model.mlp import MLP
from utils.util import ensure_dir

if __name__ == '__main__':
    #############################################
    #### MLP on inseparable two class data  #####
    #############################################

    save_folder = "../../imgs"
    # eta_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.05, 0.1, 0.25, 1, 10, 100]
    eta_values = [0.002, 0.005, 0.01, 0.05, 0.1, 0.25]
    epochs = 100000  # max number of epochs
    early_stop = 150
    shuffle = False
    loops = 4

    dataset, train_sets, valid_sets = perpare_reproducable_inseparable_dataset_2_with_subsets()
    train_sets.append(dataset) # To have the original dataset with train=valid (aka no validation)
    valid_sets.append(dataset)
    ensure_dir(save_folder)

    for dataset_idx in [0,1,2,3,4]:
        train = (train_sets[dataset_idx][0].T.copy(), train_sets[dataset_idx][1].T.copy())
        valid = (valid_sets[dataset_idx][0].T.copy(), valid_sets[dataset_idx][1].T.copy())
        train[1][train[1] == -1] = 0 # cause of sigmoid output
        valid[1][valid[1] == -1] = 0
        batch_size = train[0].shape[0]

        print(f"{'#' * 60}\n{'#' * 29}{dataset_idx:^3}{'#' * 28}\n{'#' * 60}")

        results = {}
        metrics = [
            "pocket_epoch",
            "train_loss", "train_acc", "train_sens", "train_spec",
            "valid_loss", "valid_acc", "valid_sens", "valid_spec"
        ]
        for n_hidden_nodes in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print(f"x-----> {n_hidden_nodes} hidden nodes <-----x")
            results[n_hidden_nodes] = {}
            for eta in eta_values:
                print(f"ETA is {eta}")
                results[n_hidden_nodes][eta] = {}
                accumulated_metrics = {}
                for m in metrics:
                    accumulated_metrics[m] = []

                name = f"MLP_1_d:{dataset_idx}_b:{batch_size}_h:{n_hidden_nodes}_eta:{eta}_".replace(".", ",")
                print(name)
                for _ in range(loops):
                    print(".", end="")

                    net = MLP(train[0], train[1], n_hidden_nodes, momentum=0)
                    # net.train_for_niterations(train[0], train[1], eta, max_iter)
                    train_losses, valid_losses, train_accuracies, valid_accuracies, pocket_epoch = net.train(
                        train[0], train[1], valid[0], valid[1], eta, epochs,
                        early_stop_count=early_stop, shuffle=shuffle, batch_size=batch_size
                    )
                    cm_train = net.confmat(train[0], train[1])
                    cm_valid = net.confmat(valid[0], valid[1])
                    # print(train_losses, valid_losses, train_accuracies, valid_accuracies, pocket_epoch)
                    print(cm_train)
                    print(cm_valid)

                    train_loss, valid_loss = train_losses[pocket_epoch], valid_losses[pocket_epoch]
                    train_acc, train_sens, train_spec, _, _ = two_class_conf_mat_metrics(
                        net.confmat(train[0], train[1]))
                    valid_acc, valid_sens, valid_spec, _, _ = two_class_conf_mat_metrics(
                        net.confmat(valid[0], valid[1]))

                    for m in metrics:
                        accumulated_metrics[m].append(locals()[m])
                print()
                for m in metrics:
                    results[n_hidden_nodes][eta][m] = np.array(accumulated_metrics[m]).mean(), np.array(
                        accumulated_metrics[m]).std()
            print(results)
            print(".\n.\n.")

        for k, v in results.items():
            print(f"n_hidden:{k}")
            print_results_as_table(results[k], metrics)
