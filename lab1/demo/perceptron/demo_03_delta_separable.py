import numpy as np

from demo.perceptron.demo_util import delta_rule_learning_demo, \
    perpare_reproducable_separable_dataset, print_results_as_table
from utils.util import ensure_dir

if __name__ == '__main__':
    ######################
    #### DELTA. RULE #####
    ######################

    save_folder = "../../imgs"
    eta_values = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.1, 0.2, 0.3]
    max_iter = 10000  # max number of epochs
    debug = True
    delta_n = 50  # number of epochs without improvements in delta learning
    delta_n_batch = 150
    bias = True
    loops = 1  # 100
    inputs, targets = perpare_reproducable_separable_dataset()

    ensure_dir(save_folder)
    delta_results = {}
    for batch_size in [targets.shape[1], 1]:
        delta_results[batch_size] = {}
        for eta in eta_values:
            delta_results[batch_size][eta] = {}
            print(f"ETA is {eta}")

            accuracies = []
            losses = []
            convergence_epochs = []
            for i in range(loops):
                acc, loss, cepoch = delta_rule_learning_demo(inputs, targets,
                                                             f"DELTA_RULE{'' if bias else '_NO_BIAS'}_b:{batch_size}_eta:{eta}_max_iter:{max_iter}_i:{i}".replace(
                                                                 ".", ","), debug and i == 0, save_folder, max_iter,
                                                             eta, (delta_n if batch_size == 1 else delta_n_batch),
                                                             batch_size, bias)
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

""" 
DELTA RULE
batch_size > eta > results {
    1: {
        0.0001: {
            'accuracy': (1.0, 0.0),
            'loss': (5.1245950868735924e-05, 0.00015750512790926345),
            'co_epoch': (80.0, 40.98755908809404)
        },
        0.001: {
            'accuracy': (1.0, 0.0),
            'loss': (4.550774610174047e-05, 7.549173555675504e-05),
            'co_epoch': (40.73, 27.370369014684474)
        },
        0.002: {
            'accuracy': (1.0, 0.0),
            'loss': (3.49331870579021e-05, 6.571903521876009e-05),
            'co_epoch': (45.93, 30.77279805282581)
        },
        0.003: {
            'accuracy': (1.0, 0.0),
            'loss': (4.032476368822446e-05, 8.010552283793348e-05),
            'co_epoch': (43.78, 31.402095471480877)
        },
        0.004: {
            'accuracy': (1.0, 0.0),
            'loss': (3.187054643239433e-05, 5.720921071144335e-05),
            'co_epoch': (42.24, 35.72873353478961)
        },
        0.005: {
            'accuracy': (1.0, 0.0),
            'loss': (3.0828295926006e-05, 4.7085906377854255e-05),
            'co_epoch': (42.13, 31.520677340437974)
        },
        0.01: {
            'accuracy': (1.0, 0.0),
            'loss': (1.942957814856758e-05, 4.8788364234587527e-05),
            'co_epoch': (42.67, 30.2046536149647)
        },
        0.1: {
            'accuracy': (1.0, 0.0),
            'loss': (4.454485442030998e-05, 0.00011561593460809949),
            'co_epoch': (40.72, 31.781151646848798)
        },
        0.2: {
            'accuracy': (1.0, 0.0),
            'loss': (0.01717124638615387, 0.02902762702904672),
            'co_epoch': (41.84, 29.45156023031717)
        },
        0.3: {
            'accuracy': (0.9048499999999998, 0.015466980959450363),
            'loss': (1.097804899674457, 2.5911591472547446),
            'co_epoch': (39.79, 32.62216884267507)
        }
    },
    200: {
        0.0001: {
            'accuracy': (1.0, 0.0),
            'loss': (0.058141321725386046, 3.5441613195208363e-07),
            'co_epoch': (1081.97, 245.29877517019932)
        },
        0.001: {
            'accuracy': (1.0, 0.0),
            'loss': (0.058140902212747536, 1.5112867965627075e-07),
            'co_epoch': (110.99, 25.0181913814728)
        },
        0.002: {
            'accuracy': (1.0, 0.0),
            'loss': (0.058141323316140754, 7.673993549109264e-08),
            'co_epoch': (117.69, 16.386393746032102)
        },
        0.003: {
            'accuracy': (1.0, 0.0),
            'loss': (8.209586102499784, 9.889236636295827),
            'co_epoch': (2.1, 0.6557438524302001)
        },
        0.004: {
            'accuracy': (1.0, 0.0),
            'loss': (18.400528322804217, 27.261147211289618),
            'co_epoch': (1.92, 0.4166533331199932)
        },
        0.005: {
            'accuracy': (1.0, 0.0),
            'loss': (33.531465330682956, 50.83120182639493),
            'co_epoch': (1.91, 0.4493328387732194)
        },
        0.01: {
            'accuracy': (1.0, 0.0),
            'loss': (975.8090242073113, 5597.034824033191),
            'co_epoch': (2.07, 0.5148786264742401)
        },
        0.1: {
            'accuracy': (1.0, 0.0),
            'loss': (3140212950979.994, 22651402941566.816),
            'co_epoch': (2.04, 0.7863841300535)
        },
        0.2: {
            'accuracy': (1.0, 0.0),
            'loss': (8517931279608800.0, 8.384522198948566e+16),
            'co_epoch': (2.11, 0.8472897969408106)
        },
        0.3: {
            'accuracy': (1.0, 0.0),
            'loss': (2783716589588300.0, 1.9385738519863116e+16),
            'co_epoch': (2.32, 0.8818163074019442)
        }
    }
}

B:1
0.0001	0.001	0.002	0.003	0.004	0.005	0.01	0.1	0.2	0.3	
1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	0.9048499999999998	
0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.015466980959450363	
80.0	40.73	45.93	43.78	42.24	42.13	42.67	40.72	41.84	39.79	
40.98755908809404	27.370369014684474	30.77279805282581	31.402095471480877	35.72873353478961	31.520677340437974	30.2046536149647	31.781151646848798	29.45156023031717	32.62216884267507	
5.1245950868735924e-05	4.550774610174047e-05	3.49331870579021e-05	4.032476368822446e-05	3.187054643239433e-05	3.0828295926006e-05	1.942957814856758e-05	4.454485442030998e-05	0.01717124638615387	1.097804899674457	
0.00015750512790926345	7.549173555675504e-05	6.571903521876009e-05	8.010552283793348e-05	5.720921071144335e-05	4.7085906377854255e-05	4.8788364234587527e-05	0.00011561593460809949	0.02902762702904672	2.5911591472547446	
B:200
0.0001	0.001	0.002	0.003	0.004	0.005	0.01	0.1	0.2	0.3	
1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	
0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	
1081.97	110.99	117.69	2.1	1.92	1.91	2.07	2.04	2.11	2.32	
245.29877517019932	25.0181913814728	16.386393746032102	0.6557438524302001	0.4166533331199932	0.4493328387732194	0.5148786264742401	0.7863841300535	0.8472897969408106	0.8818163074019442	
0.058141321725386046	0.058140902212747536	0.058141323316140754	8.209586102499784	18.400528322804217	33.531465330682956	975.8090242073113	3140212950979.994	8517931279608800.0	2783716589588300.0	
3.5441613195208363e-07	1.5112867965627075e-07	7.673993549109264e-08	9.889236636295827	27.261147211289618	50.83120182639493	5597.034824033191	22651402941566.816	8.384522198948566e+16	1.9385738519863116e+16	
"""
