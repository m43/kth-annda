import math

import numpy as np
from tqdm import tqdm

from utils.util import rbf, mae


class RBF:
    """
    A class that represents a radial basis function (RBF) network. The network consists of a RBF layer followed by
    a single linear layer. The RBF layer can be initialized in various ways.
    """

    def __init__(self, n_of_rbfs, n_features, n_outputs, rbf_init=None, rbf_init_data=None, silent=True,
                 normalize_hidden_outputs=False):
        self.n_features = n_features
        self.n_rbfs = n_of_rbfs
        self.n_targets = n_outputs
        self.silent = silent
        self.normalize_hidden_outputs = normalize_hidden_outputs

        self.rbf_weights = None
        self.rbf_variances = None

        if rbf_init is None:
            self._random_rbf_initialization()
            if not silent: print('RBF model initialized randomly, no initialization method stated.')
        elif rbf_init == 'uniform':
            self._uniform_rfb_init(rbf_init_data)
        elif rbf_init == "kmeans":
            self._kmeans_rbf_init(rbf_init_data)
        else:
            raise Exception(f"rbf_init={rbf_init} is not supported.")

        self.slp_weights = None
        self._random_weight_initialization()

    def _random_weight_initialization(self, mean=0, stddev=1):
        """
        (Re)Initializes the weight column of a RBF network picking weights randomly from a Gaussian distribution.

        :param mean: mean value of the Gaussian distribution
        :param stddev: the standard deviation of the Gaussian distribution (non-negative)
        """
        self.slp_weights = np.random.normal(mean, stddev, size=(self.n_rbfs, self.n_targets))

    def _uniform_rfb_init(self, rbf_init_data):
        try:
            start = rbf_init_data[0]
            end = rbf_init_data[1]
            variance = None
            if len(rbf_init_data) > 2:
                variance = rbf_init_data[2]
        except RuntimeError:
            raise RuntimeError(
                f'Uniform RBF initialization error: rbf_init_data ({rbf_init_data}) '
                f'wrong, needs to have three elements, start, end and variance.')

        if variance is not None and variance != 0:
            self._uniform_rbf_initialization(start=start, end=end, variance=variance)
        else:
            self._uniform_rbf_initialization(start=start, end=end)

    def _random_rbf_initialization(self, start=0, end=2 * math.pi, variance=0.5):
        self.rbf_variances = np.ones(self.n_rbfs) * variance
        self.rbf_weights = np.random.uniform(low=start, high=end, size=(self.n_features, self.n_rbfs))

    def _uniform_rbf_initialization(self, start, end, variance=None):
        distance = abs((end - start) / 2) if self.n_rbfs == 1 else abs((end - start) / (self.n_rbfs - 1))
        self.rbf_variances = np.ones(self.n_rbfs) * (variance if variance else math.sqrt(distance))
        self.rbf_weights = np.array(
            [[start + i * distance for i in range(self.n_rbfs)] for _ in range(self.n_features)])

    def _kmeans_rbf_init(self, rbf_init_data):
        try:
            inputs = rbf_init_data["inputs"]
            kmeans = rbf_init_data["kmeans"]
            eta = rbf_init_data.get("eta", 0.1)
            n_iter = rbf_init_data.get("n_iter", 1000)
            variance_rescale = rbf_init_data.get("variance_rescale", (0, 1))
        except RuntimeError:
            raise RuntimeError(
                f'Kmeans RBF initialization error: rbf_init_data ({rbf_init_data}) '
                f'wrong, rbf_init_data needs to be an dictionary with:'
                f'key:"inputs" - inputs to cluster'
                f'key:"kmeans" - an instance of KMeans class'
                f'optional key:"eta" - learning rate for kmeans'
                f'optional key:"n_iter" - n of iterations for KMeans fitting'
                f'optional:"variance_rescale" - tuple (add_to_all, scale_by), new_variances=(old+add_to_all)*scale_by')

        kmeans.fit(inputs, eta, n_iter)
        self.rbf_weights = kmeans.get_weights()
        self.rbf_variances = (kmeans.cluster_variances(inputs) + variance_rescale[0]) * variance_rescale[1]

    def _rbf_forward(self, inputs):
        if len(inputs.shape) == 1:
            inputs = inputs[np.newaxis]

        if inputs.shape[1] != self.n_features:
            raise Exception(f"Invalid inputs shape {inputs.shape}")

        self.rbf_outputs = np.array(
            [[rbf(single_input, self.rbf_weights[:, i].T, self.rbf_variances[i]) for i in range(self.n_rbfs)
              ] for single_input in inputs])  # TODO write in one line :)

        if self.normalize_hidden_outputs:
            self.rbf_outputs = self.rbf_outputs / self.rbf_outputs.sum(axis=1).reshape(-1, 1)

    def _slp_forward(self):
        self.slp_outputs = self.rbf_outputs @ self.slp_weights

    def least_squares_training(self, inputs, targets):
        """
        Changes weights according to the least squares method.

        :param inputs: input data
        :param targets: target data
        """
        self._rbf_forward(inputs)
        a = self.rbf_outputs.T @ self.rbf_outputs
        b = self.rbf_outputs.T @ targets
        self.slp_weights = np.linalg.solve(a, b)

    def delta_learning(self, inputs, targets, eta, val_inputs=None, val_targets=None,
                       early_stop_threshold=10e-5, early_stop_tolerance=100, epochs=10000):
        """
        Changes weights according to the on-line delta rule.

        :param inputs: training inputs
        :param targets: training targets
        :param val_inputs: validation inputs for early stopping (if None training is used for stopping)
        :param val_targets: validation targets for early stopping (if None training is used for stopping)
        :param eta: learning rate, eta of the delta rule
        :param early_stop_threshold: the threshold value for improvement in the early stopping technique
        :param early_stop_tolerance: allowed number of iterations without improvement
        :return: the best MAE on the validation set and the number of iterations it took to converge
        """
        pocket_mae = float('inf')
        pocket_epoch = 0
        pocket_weights = None
        no_improvement = 0

        for epoch in tqdm(range(1, epochs + 1)):
            self._rbf_forward(inputs)

            indices = np.arange(self.rbf_outputs.shape[0])
            np.random.shuffle(indices)

            # one epoch
            for i in indices:
                output = self.rbf_outputs[i:i + 1] @ self.slp_weights
                self.slp_weights += eta * (targets[i:i + 1] - output) * self.rbf_outputs[i:i + 1].T

            # use validation for early stopping if it is available
            if val_inputs is not None and val_targets is not None:
                error = self.evaluate_mae(val_inputs, val_targets)
            else:
                error = self.evaluate_mae(inputs, targets)

            if error + early_stop_threshold < pocket_mae:
                no_improvement = 0
                pocket_mae = error
                pocket_epoch = epoch
                pocket_weights = self.slp_weights.copy()
            else:
                no_improvement += 1
                if no_improvement > early_stop_tolerance:
                    break

        self.slp_weights = pocket_weights
        return pocket_mae, pocket_epoch

    def forward_pass(self, inputs):
        """
        Calculates the output of the network for given inputs.

        :param inputs: (n_of_inputs, n_features) inputs matrix
        :return: (n_of_inputs, n_targets) outputs matrix
        """
        self._rbf_forward(inputs)
        self._slp_forward()
        return self.slp_outputs

    def evaluate_mae(self, inputs, targets, transform_function=None):
        """
        Evaluates the MAE of the RBF network. Can transform outputs if a function is given.

        :param inputs: input data for the network
        :param targets: target data
        :param transform_function: optional function to use on output data
        :return: return MAE of the network's output compared to the target data
        """
        outputs = self.forward_pass(inputs)
        if transform_function:
            outputs = transform_function(outputs)
        return mae(outputs, targets)
