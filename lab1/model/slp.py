import numpy as np

from utils.util import shuffle_two_arrays, extend_inputs_with_bias


class SingleLayerPerceptron:
    def __init__(self, inputs, targets, debug=False):
        self.W = np.random.randn(targets.shape[0], inputs.shape[0] + 1) * 0.7
        self.debug = debug

    def train(self, inputs, targets, eta, minibatch_size=1, max_iter=-1, early_stopping=3, shuffle=True,
              delta=True):
        # delta_best = float('inf')
        # delta_counter = 0
        # correct = 0  # counter of correct classifications
        inputs = extend_inputs_with_bias(inputs, features_axis=0)

        counter = 0
        while True:
            if 0 < max_iter < counter:
                print('Number of possible epochs exceeded - training terminated')
                break
            counter += 1

            # shuffle after each epoch
            if shuffle:
                inputs, targets = shuffle_two_arrays(inputs, targets)

            for batch_idx in range(np.math.ceil(inputs.shape[-1] / minibatch_size)):
                x = inputs[:, batch_idx * minibatch_size:(batch_idx + 1) * minibatch_size]
                t = targets[:, batch_idx * minibatch_size:(batch_idx + 1) * minibatch_size]

                outputs = np.matmul(self.W, x)
                threshold_outputs = np.sign(outputs)
                if delta:
                    delta_weight = -eta * np.dot(outputs - t, x.T)
                else:
                    delta_weight = -eta * np.matmul(threshold_outputs - t, x.T)
                self.W += delta_weight

                ## TODO impl stopping criteria
                # if not delta and np.equal(threshold_outputs, t).all():
                #     break
                # else:
                #     if np.sum(outputs - t) < delta_best:
                #         delta_counter = 0
                #         delta_best = np.sum(outputs - t)
                #     else:
                #         delta_counter += 1
                #
                #     if delta_counter > early_stopping:
                #         break

        self.log('Converged after', counter, 'epochs.')

    def log(self, *a, **b):
        if self.debug:
            print(a, b)

    def eval(self, inputs, targets):
        # TODO
        pass

    def confmat(self, inputs, targets):
        # TODO
        pass
