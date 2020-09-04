import os
import time

import numpy as np

from model.delta_rule_perceptron import DeltaRulePerceptron
from model.perceptron_rule_perceptron import PerceptronRulePerceptron
from utils.util import TwoClassDatasetGenerator, scatter_plot_2d_features, ensure_dir


def perceptron_learning_demo(inputs, targets, name, debug, save_folder, max_iter, eta):
    if debug:
        print(f"{name} started")

    perceptron = PerceptronRulePerceptron(inputs, targets, debug, os.path.join(save_folder, name))
    if debug:
        print(f"Before training - Acc: {perceptron.eval(inputs, targets)}")
        scatter_plot_2d_features(inputs, targets, name + "_START", save_folder, perceptron.W)

    weights_per_epoch, acc, convergence_epoch = perceptron.train(inputs, targets, eta, max_iter=max_iter, shuffle=True)

    if debug:
        print(f"After training - Acc: {acc} CEpoch: {convergence_epoch}")
        scatter_plot_2d_features(inputs, targets, name + "_END", save_folder, perceptron.W)
        # TODO animate weights_per_epoch...
        print()

    return acc, convergence_epoch


def delta_rule_learning_demo(inputs, targets, name, debug, save_folder, max_iter, eta, delta_n, batch_size):
    if debug:
        print(f"{name} started")

    perceptron = DeltaRulePerceptron(inputs, targets, debug, os.path.join(save_folder, name))
    if debug:
        acc, loss = perceptron.eval(inputs, targets)
        print(f"Before training - Acc: {acc} Loss: {loss}")
        scatter_plot_2d_features(inputs, targets, name + "_START", save_folder, perceptron.W)

    weights_per_epoch, acc, loss, convergence_epoch = perceptron.train(inputs, targets, eta, max_iter, batch_size,
                                                                       shuffle=True, stop_after=delta_n)

    if debug:
        print(f"After training - Acc: {acc} Loss: {loss} CEpoch: {convergence_epoch}")
        scatter_plot_2d_features(inputs, targets, name + "_END", save_folder, perceptron.W)
        # TODO animate weights_per_epoch...
        print()

    return acc, loss, convergence_epoch


if __name__ == '__main__':
    ########################
    #### CONFIGURATION #####
    ########################
    save_folder = "../../imgs"
    # eta = 0.001  # learning rate
    max_iter = 10000  # max number of epochs
    debug = False
    delta_n = 50  # number of epochs without improvements in delta learning
    delta_n_batch = 150
    ####################
    #### DATA PREP #####
    ####################
    ensure_dir(save_folder)
    inputs, targets = TwoClassDatasetGenerator(
        m_a=(1.5, 1.5), n_a=100, sigma_a=(0.5, 0.5),
        m_b=(-1.5, -1.5), n_b=100, sigma_b=(0.5, 0.5)
    ).random(seed=72)
    # scatter_plot_2d_features(inputs, targets, "samples", save_folder)


    # batch_size, eta, debug = 200, 0.001, True
    # delta_rule_learning_demo(
    #     inputs, targets, f"DELTA_RULE_b:{batch_size}_eta:{eta}_max_iter:{max_iter}".replace(".", ","),
    #     debug, save_folder, max_iter, eta, (delta_n if batch_size == 1 else delta_n_batch), batch_size)
    # time.sleep(1000000000000000000)

    ####################
    #### PCN. RULE #####
    ####################
    pcn_results = {}
    for eta in [0.001, 0.005, 0.01, 0.1, 0.25, 1, 10, 25, 100, 1e3, 1e4, 1e5, 1e6, 1e7]:
        pcn_results[eta] = {}
        print(f"ETA is {eta}")

        accuracies = []
        convergence_epochs = []
        for i in range(10):
            acc, cepoch = perceptron_learning_demo(
                inputs, targets, f"PCN.RULE_eta:{eta}_max_iter:{max_iter}_i:{i}".replace(".", ","),
                debug and i==0, save_folder, max_iter, eta)
            cepoch += 1  # cepoch=0 means that it was the end of the first epoch, but we want this to be noted as 1
            accuracies.append(acc)
            convergence_epochs.append(cepoch)
            print(".", end="")

        m_acc, std_acc = np.array(accuracies).mean(), np.array(accuracies).std()
        m_cepoch, std_cepoch = np.array(convergence_epochs).mean(), np.array(convergence_epochs).std()
        pcn_results[eta]["accuracy"] = (m_acc, std_acc)
        pcn_results[eta]["co_epoch"] = (m_cepoch, std_cepoch)

        print("accuracies", accuracies)
        print("convergence_epochs", convergence_epochs)
        print(f"acc_mean={m_acc} acc_std={std_acc}")
        print(f"c_epoch_mean={m_cepoch} c_epoch_std={std_cepoch}")
        print()
        print()

    ######################
    #### DELTA. RULE #####
    ######################
    delta_results = {}
    for batch_size in [1, targets.shape[1]]:
        delta_results[batch_size] = {}
        for eta in [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.1]:
            delta_results[batch_size][eta] = {}
            print(f"ETA is {eta}")

            accuracies = []
            losses = []
            convergence_epochs = []
            for i in range(10):
                acc, loss, cepoch = delta_rule_learning_demo(
                    inputs, targets, f"DELTA_RULE_b:{batch_size}_eta:{eta}_max_iter:{max_iter}_i:{i}".replace(".", ","),
                    debug and i==0, save_folder, max_iter, eta, (delta_n if batch_size==1 else delta_n_batch), batch_size)
                cepoch += 1  # cepoch=0 means that it was the end of the first epoch, but we want this to be noted as 1
                accuracies.append(acc)
                losses.append(loss)
                convergence_epochs.append(cepoch)

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

    print("PCN")
    print("eta > results")
    print(pcn_results)
    print("DELTA RULE")
    print("batch_size > eta > results")
    print(delta_results)

"""
PCN
eta > results {
    0.001: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (36.4, 36.494383129462534)
    },
    0.005: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (9.3, 7.335529974037323)
    },
    0.01: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (6.2, 4.214261501141095)
    },
    0.1: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (1.8, 0.4)
    },
    0.25: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (1.7, 0.45825756949558405)
    },
    1: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (2.0, 0.0)
    },
    10: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (2.0, 0.4472135954999579)
    },
    25: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (2.0, 0.0)
    },
    100: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (1.9, 0.3)
    },
    1000.0: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (1.9, 0.3)
    },
    10000.0: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (1.7, 0.45825756949558405)
    },
    100000.0: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (2.1, 0.3)
    },
    1000000.0: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (1.6, 0.48989794855663565)
    },
    10000000.0: {
        'accuracy': (1.0, 0.0),
        'co_epoch': (1.7, 0.45825756949558394)
    }
}
DELTA RULE
batch_size > eta > results {
    1: {
        0.0001: {
            'accuracy': (1.0, 0.0),
            'loss': (1.894966509942782e-05, 1.754575587586114e-05),
            'co_epoch': (70.3, 30.825476476447207)
        },
        0.001: {
            'accuracy': (1.0, 0.0),
            'loss': (3.5327061354110556e-05, 2.7041699273818487e-05),
            'co_epoch': (36.3, 18.80452073305778)
        },
        0.002: {
            'accuracy': (1.0, 0.0),
            'loss': (3.265934717403029e-05, 4.964096905173829e-05),
            'co_epoch': (43.5, 29.994166099426735)
        },
        0.003: {
            'accuracy': (1.0, 0.0),
            'loss': (1.8040151529229665e-05, 2.5833139879044306e-05),
            'co_epoch': (38.9, 18.732058082335747)
        },
        0.004: {
            'accuracy': (1.0, 0.0),
            'loss': (1.6918349469113864e-05, 9.47990931614541e-06),
            'co_epoch': (45.8, 24.563387388550463)
        },
        0.005: {
            'accuracy': (1.0, 0.0),
            'loss': (1.1285876063378499e-05, 1.7834799136590546e-05),
            'co_epoch': (47.2, 29.002758489495445)
        },
        0.01: {
            'accuracy': (1.0, 0.0),
            'loss': (1.7437027473288048e-05, 4.075554007267446e-05),
            'co_epoch': (56.3, 23.774145620820953)
        },
        0.1: {
            'accuracy': (1.0, 0.0),
            'loss': (1.9209029752785984e-05, 2.0253493185332245e-05),
            'co_epoch': (48.7, 35.71008260981764)
        }
    },
    200: {
        0.0001: {
            'accuracy': (1.0, 0.0),
            'loss': (0.058141208737095364, 3.1539945335383727e-07),
            'co_epoch': (1059.6, 204.00990172048023)
        },
        0.001: {
            'accuracy': (1.0, 0.0),
            'loss': (0.058141018456425296, 1.531964613486476e-07),
            'co_epoch': (103.4, 27.925615481131295)
        },
        0.002: {
            'accuracy': (1.0, 0.0),
            'loss': (0.058141306976537174, 6.453790735466773e-08),
            'co_epoch': (120.8, 12.480384609458156)
        },
        0.003: {
            'accuracy': (1.0, 0.0),
            'loss': (8.56385678417402, 11.487811480351754),
            'co_epoch': (2.0, 0.8944271909999159)
        },
        0.004: {
            'accuracy': (1.0, 0.0),
            'loss': (20.331456644978694, 13.427680953449375),
            'co_epoch': (1.9, 0.3)
        },
        0.005: {
            'accuracy': (1.0, 0.0),
            'loss': (65.42375207398246, 66.50725822921733),
            'co_epoch': (2.1, 0.3)
        },
        0.01: {
            'accuracy': (1.0, 0.0),
            'loss': (156.4853941418691, 201.921871000402),
            'co_epoch': (2.0, 0.4472135954999579)
        },
        0.1: {
            'accuracy': (1.0, 0.0),
            'loss': (26893927133.445255, 80681763575.06421),
            'co_epoch': (1.8, 0.8717797887081348)
        }
    }
}
"""
