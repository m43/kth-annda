import csv
import numpy as np

from model.kmeans import KMeansEuclidean, KMeansAngular

IRIS_DATASET_LOCATION = "../../datasets/iris.data"
from sklearn.cluster import KMeans

if __name__ == '__main__':
    data = []
    with open(IRIS_DATASET_LOCATION) as iris_file:
        reader = csv.reader(iris_file, delimiter=',', quotechar='|')
        class_counter = 0
        class_mappings = {}
        for row in reader:
            if not row:
                continue
            class_name = row[-1]
            if class_name not in class_mappings:
                class_mappings[class_name] = class_counter
                class_counter += 1
            row[-1] = class_mappings[class_name]
            data.append([float(x) for x in row])
    data = np.array(data)

    # np.random.shuffle(data)
    inputs = data[:, :4]
    targets = data[:, 4].astype("int")

    # Data normalization
    # inputs = (inputs - inputs.mean(0))
    # inputs /= np.abs(inputs).max(0)
    # print(np.abs(inputs).max(0))

    # Targets to one hoot
    # targets = number_to_one_hoot(targets, 3)

    train = (inputs[::2], targets[::2])
    valid = (inputs[1::4], targets[1::4])
    test = (inputs[3::4], targets[3::4])

    print(f"{'-' * 12} Class: sklearn.cluster.KMeans {'-' * 12}")
    for i in range(3):
        predict = KMeans(n_clusters=3, random_state=0).fit(inputs).labels_
        n = len(predict)
        print(f"Clusters:\n{predict[:n // 3]}\n{predict[n // 3:2 * n // 3]}\n{predict[2 * n // 3:]}")
        # print(f"Targets:\n{targets[:n // 3]}\n{targets[n // 3:2 * n // 3]}\n{targets[2 * n // 3:]}")
        print(f"Cluster_SUM={np.sum(predict)} Targets_SUM={np.sum(targets)}")
        print()
    print()
    print()
    print()

    for kmeans, name in [
        (KMeansEuclidean(3, inputs.shape[1], silent=False, one_winner=True, init_from_data=False), "e-one-random"),
        (KMeansEuclidean(3, inputs.shape[1], silent=False, one_winner=False), "e-more-winners"),
        (KMeansEuclidean(3, inputs.shape[1], silent=False, one_winner=True), "e-one"),
        (KMeansAngular(3, inputs.shape[1], silent=False), "angular")
    ]:
        print(f"{'-' * 12} Class: {name} {'-' * 12}")
        for i in range(3):
            kmeans.fit(inputs, 0.1, 1000)
            predict = kmeans.predict(inputs)
            n = len(predict)
            print(f"Clusters:\n{predict[:n // 3]}\n{predict[n // 3:2 * n // 3]}\n{predict[2 * n // 3:]}")
            # print(f"Targets:\n{targets[:n // 3]}\n{targets[n // 3:2 * n // 3]}\n{targets[2 * n // 3:]}")
            print(f"Cluster_SUM={np.sum(predict)} Targets_SUM={np.sum(targets)}")
            print()
        print()
        print()
        print()
