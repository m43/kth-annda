import sys

import numpy as np

from utils.util import mse, extend_inputs_with_bias, shuffle_two_arrays, plot_metric, accuracy


class DeltaRulePerceptron:
    def __init__(self, inputs, targets, debug=False, save_folder=None, bias=True):
        self.W = np.random.randn(targets.shape[0], inputs.shape[0] + 1) / 2
        if not bias:
            self.W[:, -1] = np.zeros_like(self.W[:, -1])
        self.debug = debug
        self.save_folder = save_folder
        self.bias = bias

    def forward(self, inputs):
        self.outputs_before_threshold = np.matmul(self.W, inputs)
        self.outputs = np.sign(self.outputs_before_threshold)

    def epoch(self, inputs, targets, eta, batch_size):
        correct = 0
        loss = 0
        for batch_idx in range(np.math.ceil(inputs.shape[-1] / batch_size)):
            x = inputs[:, batch_idx * batch_size:(batch_idx + 1) * batch_size]
            t = targets[:, batch_idx * batch_size:(batch_idx + 1) * batch_size]

            self.forward(x)
            correct += np.sum(np.all(np.equal(t, self.outputs), axis=0))

            self.W -= eta * np.dot(self.outputs_before_threshold - t, x.T)
            loss = mse(self.outputs_before_threshold, t)
            loss += loss

        return correct / targets.shape[1], loss

    def train(self, inputs, targets, eta, max_iter, batch_size, shuffle=False, stop_after=100):
        inputs = extend_inputs_with_bias(inputs, features_axis=0)

        losses = []
        accuracies = []
        weights_per_epoch = []  # Want to save the weights to draw an animation later on TODO refactor
        pocket_epoch = 0
        for epoch in range(max_iter):
            if shuffle:
                inputs, targets = shuffle_two_arrays(inputs, targets)
            weights_per_epoch.append(self.W.copy())

            acc, loss = self.epoch(inputs, targets, eta, batch_size)
            accuracies.append(acc)
            losses.append(loss)

            if acc > accuracies[pocket_epoch] or acc == accuracies[pocket_epoch] and loss + 1e-6 < losses[pocket_epoch]:
                pocket_epoch = epoch  # TODO 1e-6 might be too big...
            else:
                if (epoch - pocket_epoch) > stop_after:
                    break

        self.W = weights_per_epoch[pocket_epoch]
        if self.debug:
            plot_metric(accuracies, f"{self.save_folder}_acc", True, point=(pocket_epoch, accuracies[pocket_epoch]))
            plot_metric(losses, f"{self.save_folder}_loss", True, point=(pocket_epoch, losses[pocket_epoch]))
            plot_metric(losses, f"{self.save_folder}_loss_&_acc", True, accuracies,
                        point=(pocket_epoch, losses[pocket_epoch]))
            print(f'Converged after {pocket_epoch} epochs. '
                  f'Explored {len(accuracies)} epochs. '
                  f'Maximum number of epochs was: {max_iter}.')

        return weights_per_epoch, accuracies[pocket_epoch], losses[pocket_epoch], pocket_epoch

    def eval(self, inputs, targets):
        self.forward(extend_inputs_with_bias(inputs, features_axis=0))
        acc = accuracy(targets, self.outputs, False)
        loss = mse(self.outputs_before_threshold, targets)
        return acc, loss

    def conf_mat(self, inputs, targets):
        self.forward(extend_inputs_with_bias(inputs, features_axis=0))
        loss = mse(self.outputs_before_threshold, targets)

        total = targets.shape[1]
        correct = 0

        nclasses = self.W.shape[0]
        if nclasses == 1:
            nclasses = 2
            outputs = np.where(self.outputs > 0, 1, 0)
            targets = np.where(targets > 0, 1, 0)
        else:
            # TODO has not been tested with one hoot
            outputs = np.argmax(self.outputs_before_threshold, axis=0)
            targets = np.argmax(targets, axis=0)

        cm = np.zeros((nclasses, nclasses), dtype=int)
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where((outputs == i) & (targets == j), 1, 0))
                if i == j:
                    correct += cm[i, j]

        acc = correct / total
        if self.debug:
            np.set_printoptions(threshold=sys.maxsize)
            print(cm)
            print(f"total:{total} correct:{correct}")
            print(f"acc:{acc:2f}")

        return cm
