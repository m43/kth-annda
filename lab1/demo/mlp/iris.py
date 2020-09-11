import csv

from model.mlp import MLP
from utils.util import *

IRIS_DATASET_LOCATION = "../../datasets/iris.data"

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

    np.random.shuffle(data)
    inputs = data[:, :4]
    targets = data[:, 4].astype("int")

    # Data normalization
    inputs = (inputs - inputs.mean(0))
    inputs /= np.abs(inputs).max(0)
    print(np.abs(inputs).max(0))

    # Targets to one hoot
    targets = number_to_one_hoot(targets, 3)

    train = (inputs[::2], targets[::2])
    valid = (inputs[1::4], targets[1::4])
    test = (inputs[3::4], targets[3::4])

    accs = []
    for i in range(10):
        net = MLP(train[0], train[1], 10)
        net.earlystopping_primitive(train[0], train[1], valid[0], valid[1], 0.1, 10, 2)
        # net.train(train[0], train[1], 0.1, 100000)
        print("Train")
        net.confmat(train[0], train[1])
        print("Valid")
        net.confmat(valid[0], valid[1])
        print("Test")
        acc = conf_mat_acc(net.confmat(test[0], test[1]))
        accs.append(acc)

    print(f"Mean accuracy: {np.mean(np.array(accs)) * 100:2.4f}")

"""
5. Number of Instances: 150 (50 in each of three classes)
6. Number of Attributes: 4 numeric, predictive attributes and the class
7. Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
8. Missing Attribute Values: None
9. Class Distribution: 33.3% for each of 3 classes.
Summary Statistics:
	         Min  Max   Mean    SD   Class Correlation
   sepal length: 4.3  7.9   5.84  0.83    0.7826   
    sepal width: 2.0  4.4   3.05  0.43   -0.4194
   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
    
    
RESULTS:
Without early stopping eta=0.1 and niter=10000
Confusion matrix:
[[12  0  0]
 [ 0 12  1]
 [ 0  1 11]]
Acc:94.59%

Without early stopping eta=0.1 and niter=100000
Confusion matrix:
[[10  0  0]
 [ 0 13  2]
 [ 0  0 12]]
Acc:94.59%

With early stopping eta=0.1 niter=10, Mean test accuracy: 92.4324
But it heavily depends on the concrete run, not sure why.
All test accuracies runs:
Confusion matrix:
[[ 0  0  0]
 [14 10  0]
 [ 0  0 13]]
Acc:62.16%
Confusion matrix:
[[14  0  0]
 [ 0 10  0]
 [ 0  0 13]]
Acc:100.00%
Confusion matrix:
[[14  0  0]
 [ 0 10  0]
 [ 0  0 13]]
Acc:100.00%
Confusion matrix:
[[14  0  0]
 [ 0 10  0]
 [ 0  0 13]]
Acc:100.00%
Confusion matrix:
[[14  0  0]
 [ 0 10  0]
 [ 0  0 13]]
Acc:100.00%
Confusion matrix:
[[ 0  0  0]
 [14 10  0]
 [ 0  0 13]]
Acc:62.16%
Confusion matrix:
[[14  0  0]
 [ 0 10  0]
 [ 0  0 13]]
Acc:100.00%
Confusion matrix:
[[14  0  0]
 [ 0 10  0]
 [ 0  0 13]]
Acc:100.00%
Confusion matrix:
[[14  0  0]
 [ 0 10  0]
 [ 0  0 13]]
Acc:100.00%
Confusion matrix:
[[14  0  0]
 [ 0 10  0]
 [ 0  0 13]]
Acc:100.00%


"""
