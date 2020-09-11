import numpy as np

from demo.perceptron.util import perceptron_learning_demo
from demo.util import perpare_reproducable_separable_dataset, print_results_as_table
from utils.util import ensure_dir

if __name__ == '__main__':
    ####################
    #### PCN. RULE #####
    ####################

    save_folder = "../../imgs"
    eta_values = [0.001, 0.005, 0.01, 0.1, 0.25, 1, 10, 25, 100, 1e3, 1e4, 1e5, 1e6, 1e7]
    max_iter = 10000  # max number of epochs
    debug = True
    loops = 100
    inputs, targets = perpare_reproducable_separable_dataset()

    ensure_dir(save_folder)
    pcn_results = {}
    for eta in eta_values:
        pcn_results[eta] = {}
        print(f"ETA is {eta}")

        accuracies = []
        convergence_epochs = []
        for i in range(loops):
            acc, cepoch = perceptron_learning_demo(
                inputs, targets, f"PCN.RULE_eta:{eta}_max_iter:{max_iter}_i:{i}".replace(".", ","),
                debug and i == 0, save_folder, max_iter, eta)
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

    print("PCN")
    print("eta > results")
    print(pcn_results)

    print_results_as_table(pcn_results, ["accuracy", "co_epoch"])

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
0.001	0.005	0.01	0.1	0.25	1	10	25	100	1000.0	10000.0	100000.0	1000000.0	10000000.0	
1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	
0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
29.22	8.26	3.45	1.91	1.9	1.93	1.87	1.87	1.82	1.9	1.85	1.82	1.92	1.8	
27.842622002965165	6.8009117035879845	2.1418449990603894	0.37669616403674727	0.3	0.32419130154894654	0.41605288125429435	0.36482872693909396	0.4093897898091744	0.3	0.35707142142714254	0.38418745424597084	0.3370459909270543	0.4472135954999579
"""
