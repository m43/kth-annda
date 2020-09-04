import matplotlib.pyplot as plt
import numpy as np

from model.mlp import MLP
from MLCode.Ch4.mlp import mlp

if __name__ == '__main__':
    x = np.ones((1, 40)) * np.linspace(0, 1, 40)
    t = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(40) * 0.2
    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)

    train = x[0::2]
    test = x[1::4]
    valid = x[3::4]
    traintarget = t[0::2]
    testtarget = t[1::4]
    validtarget = t[3::4]

    mean, std = train.mean(), train.std()
    print(mean, std)
    # standardize = lambda x: (x - mean) / std
    standardize = lambda x: x - mean
    # standardize = lambda x: x

    net = MLP(standardize(train), traintarget, 25, outtype="linear")
    # net.train(standardize(train), traintarget, 0.25, 100)
    net.earlystopping(standardize(train), traintarget, standardize(valid), validtarget, 0.25, 100, 2)
    net.forward(standardize(x))

    plt.plot(x, t, "b.")
    plt.plot(x, net.outputs, "r.")
    plt.show()
