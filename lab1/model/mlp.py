import numpy as np

from utils.util import extend_inputs_with_bias, sigmoid


class MLP:
    def __init__(self, inputs, targets, nhidden, beta=1, momentum=0.9, outtype='logistic'):
        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

        self.weights1 = (np.random.rand(inputs.shape[1] + 1, nhidden) - 0.5) * 2 / np.sqrt(inputs.shape[1] + 1)
        self.weights2 = (np.random.rand(nhidden + 1, targets.shape[1]) - 0.5) * 2 / np.sqrt(nhidden + 1)

        self.updatew1 = np.zeros((np.shape(self.weights1)))
        self.updatew2 = np.zeros((np.shape(self.weights2)))

    def forward(self, inputs):
        self.hidden = np.dot(extend_inputs_with_bias(inputs), self.weights1)
        self.hidden = sigmoid(self.hidden, self.beta)
        self.hidden = extend_inputs_with_bias(self.hidden)

        self.outputs = np.dot(self.hidden, self.weights2)

        # Different types of output neurons
        if self.outtype == 'linear':
            pass
        elif self.outtype == 'logistic':
            self.outputs = sigmoid(self.outputs, self.beta)
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(self.outputs), axis=1) * np.ones((1, np.shape(self.outputs)[0]))
            self.outputs = np.transpose(np.transpose(np.exp(self.outputs)) / normalisers)
        else:
            raise Exception(f"Error - outtype '{self.outtype}' not supported")

    def backward(self, inputs, targets, eta):
        # Different types of output neurons
        if self.outtype == 'linear':
            deltao = (self.outputs - targets) / len(targets)
        elif self.outtype == 'logistic':
            deltao = (self.outputs - targets) * self.beta * self.outputs * (1.0 - self.outputs)
        elif self.outtype == 'softmax':
            deltao = (self.outputs - targets) * (self.outputs * (-self.outputs) + self.outputs) / len(targets)
        else:
            raise Exception(f"Error - outtype '{self.outtype}' not supported")

        deltah = self.hidden * self.beta * (1.0 - self.hidden) * (np.dot(deltao, np.transpose(self.weights2)))

        self.updatew1 = eta * (
            np.dot(np.transpose(extend_inputs_with_bias(inputs)), deltah[:, :-1])) + self.momentum * self.updatew1
        self.updatew2 = eta * (np.dot(np.transpose(self.hidden), deltao)) + self.momentum * self.updatew2
        self.weights1 -= self.updatew1
        self.weights2 -= self.updatew2

    def train(self, inputs, targets, eta, niterations):
        for i in range(niterations):
            self.forward(inputs)
            error = np.sum((self.outputs - targets) ** 2) / 2
            if i % 10 == 0:
                print(f"Iteration {i} error:{error:.4f}")
            self.backward(inputs, targets, eta)

    def earlystopping(self, inputs, targets, valid, validtargets, eta, niteration, early_stop_count=2,
                      early_stopping_threshold=1e-7, max_iterations=-1):

        train_losses = []
        valid_losses = []
        best_weights = None
        worse_validation_loss_counter = 0
        i = 0
        while True:
            i += 1
            if max_iterations > 0:
                if i > max_iterations:
                    break
            self.train(inputs, targets, eta, niteration)
            train_loss = np.sum((self.outputs - targets) ** 2) / 2
            train_losses.append(train_loss)

            self.forward(valid)
            valid_loss = np.sum((self.outputs - validtargets) ** 2) / 2

            print(f"{i:>4} -- train_loss:{train_loss:>10.4f}      valid_loss:{valid_loss:>10.4f}")

            if not len(valid_losses) or (
                    (valid_loss - early_stopping_threshold) < valid_losses[-1] and valid_loss != valid_losses[-1]):
                best_weights = (self.weights1, self.weights2)
                worse_validation_loss_counter = 0
            else:
                worse_validation_loss_counter += 1
                if worse_validation_loss_counter == early_stop_count:
                    self.weights1, self.weights2 = best_weights
                    break
            valid_losses.append(valid_loss)

        return train_losses, valid_losses

    def confmat(self, inputs, targets):
        self.forward(inputs)

        nclasses = targets.shape[1]
        if nclasses == 1:
            nclasses = 2
            outputs = np.where(self.outputs > 0.5, 1, 0)
        else:
            outputs = np.argmax(self.outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nclasses, nclasses), dtype=int)
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where((outputs == i) & (targets == j), 1, 0))
        acc = np.trace(cm) / np.sum(cm)

        print("Confusion matrix:")
        print(cm)
        print(f"Acc:{acc * 100:.2f}%")

        return acc
