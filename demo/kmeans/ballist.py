import numpy as np
import os

from demo.util import plot_ballist
from model.kmeans import KMeansEuclidean, KMeansAngular
from utils.util import ensure_dir

BALLIST_TRAIN_DATASET_LOCATION = "../../datasets/ballist.dat"
BALLIST_TEST_DATASET_LOCATION = "../../datasets/balltest.dat"

if __name__ == '__main__':
    save_folder = "../../imgs/kmeans/ballist_clean"
    ensure_dir(save_folder)
    show_plots = False
    n_clusters = 9

    ballist = np.loadtxt(BALLIST_TRAIN_DATASET_LOCATION)
    balltest = np.loadtxt(BALLIST_TEST_DATASET_LOCATION)
    train = ballist[:, :2], ballist[:, 2:]
    test = balltest[:, :2], balltest[:, 2:]

    for kmeans, name in [
        (KMeansEuclidean(n_clusters, train[0].shape[1], silent=False, one_winner=True), "e-one")
        ,
        (KMeansEuclidean(n_clusters, train[0].shape[1], silent=False, one_winner=True, init_from_data=False),
         "e-one-random"),
        (KMeansEuclidean(n_clusters, train[0].shape[1], silent=False, one_winner=False), "e-more-winners"),
        (KMeansAngular(n_clusters, train[0].shape[1], silent=False), "angular")
    ]:
        print(f"{'-' * 12} Class: {name} {'-' * 12}")
        for i in range(3):
            kmeans.fit(train[0], 0.1, 1000)
            predict = kmeans.predict(train[0])
            n = len(predict)
            print(f"Clusters:\n{predict[:n // 3]}\n{predict[n // 3:2 * n // 3]}\n{predict[2 * n // 3:]}\n")
            # clusters, x, y, z1, z2 = kmeans.predict(train[0]), train[0][:, 0], train[0][:, 1], train[1][:, 0], train[1][
            #                                                                                                    :, 1]
            clusters, x, y, z1, z2 = kmeans.predict(test[0]), test[0][:, 0], test[0][:, 1], test[1][:, 0], test[1][:, 1]
            centres_x, centres_y = kmeans.w[0], kmeans.w[1]
            sigmas = kmeans.cluster_variances(train[0]) ** 0.5

            plot_ballist(x, y, z1, z2, clusters, centres_x, centres_y, sigmas, show_plot=True,
                         save_filename=os.path.join(save_folder, f'ballist_{name}_c={n_clusters}_i={i}'))
