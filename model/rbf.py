import numpy as np
import math
import random

from utils.util import gaussian_exp


class Rbf:
    """
    A class that represents a radial basis function (RBF) network with a single input.
    """

    def __init__(self, hidden_layer_shape, rbf_init=None, rbf_init_data=None):
        """
        Constructs an RBF network with a single input and a given number of RBF nodes.

        :param hidden_layer_shape: number of RBF nodes.
        """
        self.hidden_layer_shape = hidden_layer_shape
        self.weights = None
        self.nodes = None

        self.random_weight_initialization()

        if rbf_init is None:
            self.random_node_initialization()
            print('RBF model initialized randomly, no initialization method stated.')
        elif rbf_init == 'uniform':
            try:
                start = rbf_init_data[0]
                end = rbf_init_data[1]
            except RuntimeError:
                raise RuntimeError(
                    f'Uniform RBF initialization error: rbf_init_data ({rbf_init_data}) '
                    f'wrong, needs to have two elements.')
            self.uniform_node_initialization(start=start, end=end)

    def random_weight_initialization(self, mean=0, scale=1):
        """
        (Re)Initializes the weight column of a RBF network picking weights randomly from a  Gaussian distribution.

        :param mean: mean value of the Gaussian distribution
        :param scale: the standard deviation of the Gaussian distribution (non-negative)
        """
        self.weights = np.array([np.random.normal(mean, scale, size=self.hidden_layer_shape)]).T

    def random_node_initialization(self):
        """
        (Re)Initializes the RBF nodes picking means randomly from a (0,1) Gaussian distribution and a fixed
        standard deviation equal to 1.
        """

        self.nodes = [(random.gauss(0, 1), 1) for i in range(self.hidden_layer_shape)]

    def uniform_node_initialization(self, start, end):
        """
        (Re)Initializes the RBF nodes by distributing their means equally on an interval.

        :param start: start of the interval
        :param end: end of the interval
        """

        if self.hidden_layer_shape == 1:
            distance = abs((end - start) / 2)
        else:
            distance = abs((end - start) / (self.hidden_layer_shape - 1))
        self.nodes = [(start + i * distance, 4 * (distance ** 2)) for i in
                      range(self.hidden_layer_shape)]

    def least_squares_training(self, inputs, targets):
        """
        Changes weights according to the least squares method.

        :param inputs: input data
        :param targets: target data
        """
        hidden_matrix = self.calculate_hidden_matrix(inputs)
        a = np.matmul(hidden_matrix.T, hidden_matrix)
        b = np.matmul(hidden_matrix.T, np.array([[target for target in targets]]).T)
        self.weights = np.linalg.solve(a, b)

    def delta_training_step(self, single_input, target, learning_rate):
        """
        Changes weights according to the on-line delta rule.

        :param single_input: the input for which the learning is applied
        :param target: the expected output
        :param learning_rate: the learning rate for the delta rule
        """
        self.weights += learning_rate * (target - self.forward_pass(single_input)) ** 2 * self.calculate_hidden_output(
            single_input) / 2

    def forward_pass(self, single_input):
        """
        Calculates the output of the network for a single given input.

        :param single_input: a single input for the RBF network
        :return: the output of the network
        """

        return float(np.matmul(self.calculate_hidden_output(single_input).T, self.weights))

    def calculate_hidden_matrix(self, inputs):
        """
        Calculates a matrix of the hidden layer outputs based on input data.

        :param inputs: an iterable object of input numbers
        :return: returns a numpy matrix with each row containing the hidden output for a single input
        """
        hidden_matrix = []
        for single_input in inputs:
            hidden_matrix.append(self.calculate_hidden_output(single_input).T[0])
        return np.array(hidden_matrix)

    def calculate_hidden_output(self, single_input):
        """
        Returns the output of the RBF nodes in the hidden layer of the RBF network for a single given input.

        :param single_input: a number for which to calculate the hidden output
        :return: hidden output as a numpy column
        """
        return np.array([[gaussian_exp(single_input, node[0], node[1]) for node in self.nodes]]).T
