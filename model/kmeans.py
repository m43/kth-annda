from abc import ABC, abstractmethod

import numpy as np

from util import normalize_vectors


class KMeans(ABC):
    @abstractmethod
    def __init__(self, k, n_features, silent=True, init_from_data=True):
        """
        :param k: the number of clusters
        :param n_features: the number of input features
        :param silent: should the network not print debug messages
        :param init_from_data: should the centers be by random data points or rather randomly in (min, max) data range
        """
        self.n_features = n_features
        self.k = k
        self.w = None
        self.silent = silent
        self.init_from_data = init_from_data

    def _initialize_centers_randomly(self, inputs):
        mini = inputs.min(axis=0).reshape(-1, 1)
        maxi = inputs.max(axis=0).reshape(-1, 1)
        self.hidden = np.random.rand(self.n_features, self.k) * (maxi - mini) + mini

    def _initialize_centers_from_inputs(self, inputs):
        self.w = inputs[np.random.choice(inputs.shape[0], self.k, replace=False)].T.copy()

    def _initialize_centers(self, inputs):
        if self.init_from_data:
            self._initialize_centers_from_inputs(inputs)
        else:
            self._initialize_centers_randomly()

    @abstractmethod
    def _determine_winners(self, inputs):
        pass

    def _forward(self, inputs):
        if self.w is None:
            raise Exception("Train the network before using it")

        if len(inputs.shape) == 1:
            inputs = inputs[np.newaxis]

        self.outputs = self._determine_winners(inputs)

    @abstractmethod
    def fit(self, inputs, eta, epochs=1000, shuffle=True):
        """
        Train the network using given inputs and learning parameters.

        :param inputs: either (n_of_examples x n_features) matrix or (n_features,) vector
        :param eta: learning rate
        :param epochs: number of epochs
        :param shuffle: should the data be shuffled before each epoch
        """
        pass

    @abstractmethod
    def predict(self, inputs):
        """
        Return the winners for each data example.

        :param inputs: either (n_of_examples x n_features) matrix input or (n_features,) vector input
        :return: (n_of_examples,) vector of winners, one for each input example
        """
        pass


class KMeansEuclidean(KMeans):
    """
    The k-Means Algorithm implemented as a neural nets. Euclidean distance measure used for clustering.
    """

    def __init__(self, k, n_features, silent=True, one_winner=True, init_from_data=True):
        super().__init__(k, n_features, silent, init_from_data)
        self.one_winner = one_winner

    def _determine_winners(self, inputs):
        self.hidden = np.sum((inputs[:, :, np.newaxis] - self.w[np.newaxis, :, :]) ** 2, axis=1)
        return np.argmin(self.hidden, axis=1)

    def fit(self, inputs, eta, epochs=1000, shuffle=True):
        self._initialize_centers_from_inputs(inputs)
        inputs = inputs.copy()  # TODO should i copy and shuffle the array or just reindex the inputs each epoch

        for epoch in range(epochs):
            if shuffle: np.random.shuffle(inputs)
            old_weights = self.w.copy()

            for i in range(len(inputs)):
                self._forward(inputs[i:i + 1])
                if self.one_winner:
                    winner = self.outputs[0]
                    self.w[:, winner] += eta * (inputs[i] - self.w[:, winner])
                else:
                    # TODO check if this is a good idea for avoiding dead units - which was the goal

                    # maybe update each according to relative activation to others?
                    # winners = normalize_vectors(1 - normalize_vectors(self.hidden))  # 1xK
                    # self.w += eta * winners * (inputs[i:i + 1].T - self.w)

                    # maybe update 2 winners?
                    for winner, factor in zip(list(reversed(np.argpartition(self.hidden[0], -2)[-2:])), [1, 0.5]):
                        self.w[:, winner] += factor * eta * (inputs[i] - self.w[:, winner])

            if np.all(old_weights == self.w):
                break

        if not self.silent: print(f"Training finished in epoch {epoch + 1} of {epochs} epochs.")
        return {"epoch": epoch}

    def predict(self, inputs):
        self._forward(inputs)
        return self.outputs


class KMeansAngular(KMeans):
    """
    The k-Means Algorithm implemented as a neural net. Angular distance measure used for clustering.
    """

    def __init__(self, k, n_features, silent=True, init_from_data=True):
        super().__init__(k, n_features, silent, init_from_data)

    def _determine_winners(self, inputs):
        return np.argmax(np.dot(inputs, self.w), axis=1)

    def fit(self, inputs, eta, epochs=1000, shuffle=True):
        self._initialize_centers_from_inputs(inputs)
        self.w = normalize_vectors(self.w, vectors_in_rows=False)

        inputs = inputs.copy()  # TODO should i copy and shuffle the array or just reindex the inputs each epoch
        inputs = normalize_vectors(inputs)

        for epoch in range(epochs):
            if shuffle: np.random.shuffle(inputs)
            old_weights = self.w.copy()
            for i in range(len(inputs)):
                self._forward(inputs[i:i + 1])
                winner = self.outputs[0]
                self.w[:, winner] += eta * (inputs[i] - self.w[:, winner])
                self.w[:, winner] = normalize_vectors(self.w[:, winner].reshape(1, -1))
            if np.all(old_weights == self.w):
                break

        if not self.silent: print(f"Training finished in epoch {epoch + 1} of {epochs} epochs.")

        return {"epoch": epoch}

    def predict(self, inputs):
        inputs = normalize_vectors(inputs)
        self._forward(inputs)
        return self.outputs
