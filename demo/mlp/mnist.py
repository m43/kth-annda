import gzip

import pickle

from model.mlp import MLP
from utils.util import number_to_one_hoot, standardize_dataset

if __name__ == '__main__':
    with gzip.open('../../datasets/mnist.pkl.gz', 'rb') as f:
        data = pickle._Unpickler(f)
        data.encoding = 'latin1'  # set encoding
        train, valid, test = data.load()

    n = 60000
    train, valid, test = (train[0][:n], train[1][:n]), (valid[0][:n], valid[1][:n]), (test[0][:n], test[1][:n])
    mean, std = train[0].mean(), train[0].std()
    train = standardize_dataset(train, mean, std)
    valid = standardize_dataset(valid, mean, std)
    test = standardize_dataset(test, mean, std)

    # for i in [1, 2, 5, 10, 20, 100]:
    for i in [20]:
        print(f"x-----> {i} hidden nodes <-----x")
        net = MLP(train[0], number_to_one_hoot(train[1]), i, outtype="softmax")
        # net.train(train[0], number_to_one_hoot(train[1]), 0.1, 1000)
        net.earlystopping_primitive(train[0], number_to_one_hoot(train[1]), valid[0], number_to_one_hoot(valid[1]),
                                    eta=0.1, niteration=100, early_stop_count=2)
        net.confmat(train[0], number_to_one_hoot(train[1]))
        net.confmat(test[0], number_to_one_hoot(test[1]))
        print()
        print()

"""
Confusion matrix:
[[4827    0   15    6   13   16   23    5   15   21]
 [   1 5541   22   12    7    8    9   12   37    8]
 [  10   25 4700   60   15    5   13   38   24   18]
 [   8   24   41 4780    2   81    2   18   51   31]
 [   6    6   40    2 4659   25   13   34   12   71]
 [  23   19   18   90    4 4260   46    7   40   27]
 [  19    3   26   10   46   31 4826    5   19    6]
 [   5   13   39   44    8    8    2 4965    8   92]
 [  30   33   51   68   10   52   17   11 4614   42]
 [   3   14   16   29   95   20    0   80   22 4672]]
Acc:95.69%
Confusion matrix:
[[ 957    0    8    1    1    5   11    2    4    7]
 [   0 1110    6    1    1    2    3    3    3    5]
 [   2    4  958    8    4    2    7   17    6    1]
 [   1    3    8  938    0   29    1   10   14   10]
 [   0    1    9    0  935    4    9    3    8   17]
 [  12    2    3   20    1  816   18    3   15   11]
 [   5    4    8    2   14   11  900    0   10    1]
 [   1    2   10    9    3    6    2  969    9   14]
 [   2    9   20   29    2   12    7    4  900   11]
 [   0    0    2    2   21    5    0   17    5  932]]
Acc:94.15%
"""
