import numpy as np
import os

from demo.util import plot_ballist
from model.kmeans import KMeansEuclidean
from model.rbf import RBF
from utils.util import ensure_dir

BALLIST_TRAIN_DATASET_LOCATION = "../../datasets/ballist.dat"
BALLIST_TEST_DATASET_LOCATION = "../../datasets/balltest.dat"

if __name__ == '__main__':
    save_folder = "../../imgs/rbf/ballist_13_final"
    ensure_dir(save_folder)
    ks = [30, 50]
    show_plots = False
    loops = 1
    # etas = [0.05, 0.25]
    etas = [0.25]
    normalize = True
    variance_rescale = (0.03, 1)

    ballist = np.loadtxt(BALLIST_TRAIN_DATASET_LOCATION)
    balltest = np.loadtxt(BALLIST_TEST_DATASET_LOCATION)

    np.random.seed(720)
    np.random.shuffle(ballist)
    train = np.concatenate((ballist[::2, :2], ballist[1::4, :2])), np.concatenate((ballist[::2, 2:], ballist[1::4, 2:]))
    valid = ballist[3::4, :2], ballist[3::4, 2:]
    test = balltest[:, :2], balltest[:, 2:]
    print(train[0].shape)
    print(valid[0].shape)

    for k in ks:
        for eta in etas:
            for kmeans, name in [
                (KMeansEuclidean(k, train[0].shape[1], silent=False, one_winner=True), "e-one")
                ,
                (KMeansEuclidean(k, train[0].shape[1], silent=False, one_winner=True, init_from_data=False),
                 "e-one-random")
            ]:
                print(f"{'-' * 12} Class: {name} {'-' * 12}")
                for i in range(loops):
                    run_name = f"ballist_{name}_k={k}_eta={eta}_i={i}_vr={variance_rescale}"
                    save_prefix = os.path.join(save_folder, f"k{k}/eta{eta}")
                    ensure_dir(save_prefix)

                    rbf = RBF(k, train[0].shape[1], train[1].shape[1], normalize_hidden_outputs=normalize,
                              rbf_init="kmeans",
                              rbf_init_data={
                                  "inputs": train[0],
                                  "kmeans": kmeans,
                                  "eta": 0.1,
                                  "n_iter": 1000,
                                  "variance_rescale": variance_rescale
                              })

                    valid_mae, pocket_epoch = rbf.delta_learning(train[0], train[1], eta, valid[0], valid[1])
                    # pocket_mae, pocket_epoch = rbf.delta_learning(train[0], train[1], eta)
                    test_mae = rbf.evaluate_mae(test[0], test[1])
                    print(f"RUN_NAME:{run_name} VALID_MAE:{valid_mae} TEST_MAE:{test_mae} P_EPOCH:{pocket_epoch}")
                    print(f"Test MAE:{test_mae}")

                    centres_x, centres_y, sigmas = kmeans.w[0], kmeans.w[1], rbf.rbf_variances ** 0.5
                    clusters, x, y, z = kmeans.predict(train[0]), train[0][:, 0], train[0][:, 1], rbf.forward_pass(
                        train[0])
                    # plot_ballist(x, y, z[:, 0], z[:, 1], clusters, centres_x, centres_y, sigmas, show_plot=show_plots,
                    #              save_filename=os.path.join(save_prefix,
                    #                                         f'{run_name}_train_testmae={test_mae}_coepoch={pocket_epoch}.png'),
                    #              title=run_name + f"_train_testmae={test_mae}_coepoch={pocket_epoch}")

                    clusters, x, y, z = kmeans.predict(test[0]), test[0][:, 0], test[0][:, 1], rbf.forward_pass(test[0])
                    plot_ballist(x, y, z[:, 0], z[:, 1], clusters, centres_x, centres_y, sigmas, show_plot=show_plots,
                                 save_filename=os.path.join(save_prefix, f'{run_name}_testmae={test_mae}.png'),
                                 title=run_name + f"_testmae={test_mae}")
