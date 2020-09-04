import numpy as np

from utils.util import plot_metric, accuracy
from utils.util import shuffle_two_arrays, extend_inputs_with_bias


class PerceptronRulePerceptron:
    def __init__(self, inputs, targets, debug=False, save_folder=None):
        self.W = np.random.randn(targets.shape[0], inputs.shape[0] + 1) / 2
        self.debug = debug
        self.save_folder = save_folder

    def forward(self, inputs):
        self.outputs = np.sign(np.matmul(self.W, inputs))

    def epoch(self, inputs, targets, eta):
        correct = 0
        for idx in range(inputs.shape[1]):
            i, t = inputs[:, idx], targets[:, idx]
            self.forward(i)
            if np.equal(self.outputs, t):
                correct += 1
            self.W -= eta * np.dot(self.outputs - t, i.reshape(1, -1))

        return correct / targets.shape[1]

    def train(self, inputs, targets, eta, max_iter, shuffle=False):
        inputs = extend_inputs_with_bias(inputs, features_axis=0)

        accuracies = []
        weights_per_epoch = []  # Want to save the weights to draw an animation later on TODO refactor
        pocket_epoch = 0
        for epoch in range(max_iter):
            weights_per_epoch.append(self.W)
            if shuffle:
                inputs, targets = shuffle_two_arrays(inputs, targets)

            acc = self.epoch(inputs, targets, eta)
            accuracies.append(acc)

            if not epoch or accuracies[pocket_epoch] < acc:
                pocket_epoch = epoch
            if acc == 1:
                break

        self.W = weights_per_epoch[pocket_epoch]
        if self.debug:
            plot_metric(accuracies, f"{self.save_folder}_acc", True,
                        point=(pocket_epoch, accuracies[pocket_epoch]))
            print(f'Converged after {pocket_epoch} epochs. Maximum number of epochs was: {max_iter}.')

        return weights_per_epoch, accuracies[pocket_epoch], pocket_epoch

    def eval(self, inputs, targets):
        self.forward(extend_inputs_with_bias(inputs, features_axis=0))
        return accuracy(inputs, targets, False)
        # print(f"Accuracy:{acc:>3.4f}")
