import numpy as np

from utils.util import shuffle_two_arrays, extend_inputs_with_bias, horse, sse


class SingleLayerPerceptron:
    def __init__(self, inputs, targets, debug=False):
        self.W = np.random.randn(targets.shape[0], inputs.shape[0] + 1) / 2
        self.debug = debug

    def forward(self, inputs):
        self.outputs_before_threshold = np.matmul(self.W, inputs)
        self.outputs = np.sign(self.outputs_before_threshold)

    def epoch(self, inputs, targets, eta, minibatch_size, delta):
        correct = 0
        self.loss = 0
        for batch_idx in range(np.math.ceil(inputs.shape[-1] / minibatch_size)):
            x = inputs[:, batch_idx * minibatch_size:(batch_idx + 1) * minibatch_size]
            t = targets[:, batch_idx * minibatch_size:(batch_idx + 1) * minibatch_size]

            self.forward(x)
            correct += np.sum(np.where(t == self.outputs, 1, 0))

            if delta:
                self.W -= eta * np.dot(self.outputs_before_threshold - t, x.T)
                self.loss += sse(self.outputs_before_threshold, targets)
            else:
                self.W -= eta * np.dot(self.outputs - t, x.T)

        self.accuracy = correct / targets.shape[1]

    def train(self, inputs, targets, eta, minibatch_size=1, max_iter=-1, early_stopping=3, shuffle=False,
              delta=True):
        inputs = extend_inputs_with_bias(inputs, features_axis=0)

        train_losses = []
        valid_losses = []  # TODO
        pocket = None
        epoch = 0
        while True:
            epoch += 1
            if 0 < max_iter < epoch:
                print(f'Maximum number ({max_iter}) of epochs exceeded - training terminated')
                break
            if shuffle:
                inputs, targets = shuffle_two_arrays(inputs, targets)
            self.epoch(inputs, targets, eta, minibatch_size, delta)

            if delta:
                train_losses.append(self.loss)
                # TODO implement early stopping using valid dataset
            else:
                # TODO how can perceptron know that it's stuck in an infinite loop? It's hard to merge perc. learn.
                #  and delta rule into one self.epoch() metod. Consider restructure.
                #  Current implementation stops if 1. max_iter reached 2. solution found (acc=100%)
                if self.accuracy == 1:
                    break

        if pocket:
            self.W = pocket

        self.log(f'Converged after {epoch} epochs.')
        # self.log(horse)  # horsified

    def log(self, *a, **b):
        if self.debug:
            for elem in a:
                print(elem)
            if b:
                print(b)

    # TODO Should I ask for delta_metrics?  How should one know what metrics are expected - loss or acc or both?
    def eval(self, inputs, targets, delta_metrics=True):
        self.forward(extend_inputs_with_bias(inputs, features_axis=0))

        acc = np.sum(np.where(targets == self.outputs, 1, 0)) * 100 / targets.shape[-1]  # TODO works only for 1d target
        print(f"Accuracy:{acc:>3.4f}")

        if delta_metrics:
            loss = sse(self.outputs_before_threshold, targets)
            print(f"Loss:{loss:>10.4f}")

    def confmat(self, inputs, targets):
        # TODO
        pass
