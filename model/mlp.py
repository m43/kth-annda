import numpy as np
from tqdm.auto import tqdm

from utils.util import extend_inputs_with_bias, sigmoid, mse


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

    def epoch(self, inputs, targets, eta, batch_size):
        correct = 0
        loss = 0

        for batch_idx in range(np.math.ceil(inputs.shape[0] / batch_size)):
            x = inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            t = targets[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            self.forward(x)
            self.backward(x, t, eta)
            if self.outtype == "logistic":  # TODO
                correct += np.sum(np.equal(t, np.where(self.outputs > 0.5, 1., 0.)))
            loss += mse(self.outputs, t)

        if self.outtype == "logistic":  # TODO
            self.acc = correct / targets.shape[0]

        return loss

    def train_for_niterations(self, inputs, targets, eta, niterations):
        for i in range(niterations):
            self.forward(inputs)
            error = np.sum((self.outputs - targets) ** 2) / 2
            # if i % 1000 == 0:
            #     print(f"Iteration {i} error:{error:.4f}")
            self.backward(inputs, targets, eta)

    def earlystopping_primitive(self, inputs, targets, valid, validtargets, eta, niteration, early_stop_count=2,
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
            self.train_for_niterations(inputs, targets, eta, niteration)
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

    def train(self, inputs, targets, valid, validtargets, eta, epochs, early_stop_count=100,
              early_stopping_threshold=1e-7, shuffle=False, batch_size=None):
        if batch_size is None:
            batch_size = targets.shape[0]

        train_accuracies = []
        valid_accuracies = []
        train_losses = []
        valid_losses = []
        pocket_epoch = 0
        pocket_weights = (self.weights1, self.weights2)
        for epoch in tqdm(range(epochs)):
            # for epoch in range(epochs):
            if shuffle:
                indices = np.arange(inputs.shape[0])
                np.random.shuffle(indices)
                inputs, targets = inputs[indices], targets[indices]

            loss = self.epoch(inputs, targets, eta, batch_size)
            train_losses.append(loss)
            if self.outtype == "logistic":  # TODO
                train_accuracies.append(self.acc)

            self.forward(valid)
            valid_losses.append(mse(self.outputs, validtargets))
            if self.outtype == "logistic":  # TODO
                valid_accuracies.append(
                    np.sum(np.equal(validtargets, np.where(self.outputs > 0.5, 1., 0.))) / validtargets.shape[0])

            if valid_losses[-1] + early_stopping_threshold < valid_losses[pocket_epoch]:
                pocket_epoch = epoch
                pocket_weights = (self.weights1, self.weights2)
            elif (epoch - pocket_epoch) > early_stop_count:
                break

        self.weights1, self.weights2 = pocket_weights

        return train_losses, valid_losses, train_accuracies, valid_accuracies, pocket_epoch

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

        return cm
