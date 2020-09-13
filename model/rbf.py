import numpy as np

from utils.util import gaussian_exp


class Rbf:
    """
    A class that represents a radial basis function (RBF) network with a single input.
    """

    def __init__(self, hidden_layer_shape):
        """
        Constructs an RBF network with a single input and a given number of RBF nodes.
        :param hidden_layer_shape: number of RBF nodes.
        """
        self.hidden_layer_shape = hidden_layer_shape
        self.weights = None
        self.nodes = None

        self.initialize_weights()
        self.initialize_nodes()

    # TODO: see initialization possibilities
    def initialize_weights(self):
        """
        (Re)Initializes the weight column of a RBF network.
        """
        self.weights = np.random.normal(size=self.hidden_layer_shape).T

    # TODO: see initialization possibilities
    def initialize_nodes(self):
        """
        (Re)Initializes the RBF nodes.
        """
        self.nodes = [(0.0, 1.0) for i in range(self.hidden_layer_shape)]

    # TODO: implement
    def least_squares_training(self, input, target):
        """
        Changes weights according to the least squares method.
        :param input: input data
        :param target: target data
        """
        pass

    # TODO: implement
    def delta_training_step(self, input, target, learning_rate):
        """
        Changes weights according to the on-line delta rule.
        :param input:
        """
        pass

    def calculate_hidden_output(self, input):
        """
        Returns the output of the RBF nodes in the hidden layer of the RBF network.
        :param input: a number for which to calculate the hidden output
        :return: hidden output as a numpy column
        """
        return np.array([gaussian_exp(input, node[0], node[1]) for node in self.nodes]).T
