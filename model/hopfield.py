import numpy as np


class Hopfield:
    """
    This class represents a Hopfield network which uses the Hebbian learning rule and has binary: {-1, 1} outputs.
    It requires the use of NumPy arrays and matrices for all of its operations.
    """

    def __init__(self, number_of_neurons):
        """
        Constructs a Hopfield network with a given number of neurons.

        :param number_of_neurons: The number of neurons the Hopfield network have.
        """
        self.number_of_neurons = number_of_neurons
        self.weights = None
        self.state = None

    def learn_patterns(self, patterns, scaling=True, self_connections=False):
        """
        Sets weights of a Hopfield network using the Hebbian one-shot rule using the given patterns. Will delete previous weights if there were any.

        :param patterns: A NumPy matrix of patterns to be learned. Each row represents a pattern. The length of each row (the number of columns) must be equal to the number of neurons in the network.
        :param scaling: An optional argument which determines if weights will be scaled with the reciprocal value of the number of patterns or not. True by default.
        :param self_connections: An optional argument which determines if neurons are connected to themselves or not. False by default.
        """
        if patterns.shape[1] != self.number_of_neurons:
            raise RuntimeError(
                f'Dimension mismatch - patterns have {patterns.shape[1]} features, '
                f'the network has {self.number_of_neurons} neurons. These two must be equal.')

        # one-shot Hebbian learning
        self.weights = np.matmul(patterns.T, patterns) / (patterns.shape[0] if scaling else 1)

        # delete self-connections
        if not self_connections:
            np.fill_diagonal(self.weights, 0)  # deletes self-connections (sets weight matrix diagonal to 0)

    def set_state(self, pattern):
        """
        Sets the current state of the network to a given pattern or state.

        :param pattern: A NumPy array representing a pattern/state which the network is set to.
        """

        self.state = pattern

    def update_step(self, batch=True):
        """
        Calculates the next state of a Hopfield network using the current state and the weight matrix.

        :param batch: Determines if the update step is done synchronously (batch) or asynchronously (sequential). True (synchronous) by default.
        :return: Current state of the Hopfield network after doing one full update.
        """
        if self.weights is None:
            raise RuntimeError(
                'Cannot update state if network has no set weights. Use learn_pattern function to set a weight matrix.'
            )
        if self.state is None:
            raise RuntimeError(
                'Cannot update state if no state is set. Use set_state function to set a state of the Hopfield network'
            )

        # synchronous update
        if batch:
            self.state = np.matmul(self.state, self.weights)
            self.state = np.where(self.state >= 0, 1, -1)  # sign function
        # asynchronous update
        else:
            random_sequence = np.array([i for i in range(self.number_of_neurons)])
            np.random.shuffle(random_sequence)
            for neuron_idx in random_sequence:
                self.state[neuron_idx] = 1 if np.matmul(self.state, self.weights.T[neuron_idx]) >= 0 else -1  # sign

        return self.state
