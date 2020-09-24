import numpy as np


class Hopfield:
    """
    This class represents a Hopfield network which uses the Hebbian learning rule and has binary: {-1, 1} outputs.
    It requires the use of NumPy arrays and matrices for all of its operations.
    """

    def __init__(self, number_of_neurons, debug_mode=False):
        """
        Constructs a Hopfield network with a given number of neurons.

        :param number_of_neurons: The number of neurons the Hopfield network have.
        :param debug_mode: Optional. Determines if the model is in debug mode or not. Debug mode has verbose printing. False by default.
        """
        self.number_of_neurons = number_of_neurons
        self.debug = debug_mode
        self.weights = None  # network weights
        self.state = None  # current output of the network's neurons

    def energy(self, x):
        """
        Computes the energy of a given pattern for our network

        :param x: The pattern to compute energy on
        """
        return -np.sum(np.multiply(self.weights, np.outer(x, x)))

    def learn_patterns(self, patterns, scaling=True, self_connections=True, imbalance=0.0):
        """
        Sets weights of a Hopfield network using the Hebbian one-shot rule using the given patterns. Will delete previous weights if there were any.

        :param patterns: A NumPy matrix of patterns to be learned. Each row represents a pattern. The length of each row (the number of columns) must be equal to the number of neurons in the network.
        :param scaling: Optional. Determines if weights will be scaled with the reciprocal value of the number of patterns or not. True by default.
        :param self_connections: Optional. Determines if neurons are connected to themselves or not. False by default.
        :param imbalance: Optional. Additional term reduced from each feature of each sample
        """
        if patterns.shape[1] != self.number_of_neurons:
            raise RuntimeError(
                f'Dimension mismatch - patterns have {patterns.shape[1]} features, '
                f'the network has {self.number_of_neurons} neurons. These two must be equal.')

        # one-shot Hebbian learning
        patterns = patterns - imbalance
        self.weights = np.matmul(patterns.T, patterns) / (patterns.shape[0] if scaling else 1)

        # delete self-connections
        if not self_connections:
            np.fill_diagonal(self.weights, 0)  # deletes self-connections (sets weight matrix diagonal to 0)

    def update_automatically(self, batch=True, update_cap=100000, step_callback=None, bias=0.0, output_bias=0.0,
                             output_scaling=1.0):
        """
        Calculates states using the update rule until convergence, oscillation (only when using synchronous updates) or reaching a defined cap.

        :param batch: Optional. Determines if updating is done synchronously (batch) or asynchronously (sequential). True (synchronous) by default.
        :param update_cap: Optional. Determines how many updates will happen are done without convergence. If the cap is reached updating stops and the function returns the last and current state. 10000 by default.
        :param step_callback: Optional. A callback function called after each step update. Arguments given are a copy of the current state and the number of update starting with 1.
        :param bias: Optional. Adds bias to updating term.
        :param output_bias: Optional. Adds a bias to all outputs.
        :param output_scaling: Optional. Multiplies output. Multiplication happens before adding output bias.
        :return: State of the Hopfield network; None if failed to converge in update_cap steps or detected oscillations in batch mode.
        """
        if self.debug:
            print('\t-----------------------------')
            print(f'\tStarting automatic updating process.\n'
                  f'\tStarting state = {self.state}\n'
                  f'\tUpdating is capped at {update_cap} steps.')
            if batch:
                print('\tUsing synchronous (batch) updating.')
            else:
                print(f'\tUsing asynchronous (sequential) updating.')
            print('\t-----------------------------')

        if not batch and step_callback is not None:
            step_callback(np.copy(self.state), 0)

        step = 0  # update step counter
        previous_states = set()  # set of previously seen states (for detection of oscillations in batch learning)
        previous_states.add(tuple(np.copy(self.state)))  # add current state to set of previously seen states
        previous_state = np.copy(self.state)  # remember previous state
        # main updating loop
        while step <= update_cap:

            # update and increase step counter
            self.update_step(batch=batch, starting_step=step * self.number_of_neurons, step_callback=step_callback,
                             bias=bias)
            step += 1

            # call callback function if defined
            if step_callback is not None and batch:
                step_callback(np.copy(self.state), step)

            # check convergence
            if (previous_state == self.state).all():
                if self.debug:
                    print(f'\tStable point reached after {step} steps, stopping learning process.')
                break
            # check oscillation
            elif batch and tuple(self.state) in previous_states:
                if self.debug:
                    print(
                        f'\tOscillation appeared while synchronously updating state (step {step}), stopping learning process.')
                self.state = None
                break
            # prepare for next update
            else:
                previous_state = np.copy(self.state)
                sequential_stable_counter = 0
                if batch:
                    previous_states.add(tuple(np.copy(self.state)))

        # return state or None for oscillations and divergence
        if self.state is None or step >= update_cap:
            return None
        return np.copy(self.state)

    def set_state(self, pattern):
        """
        Sets the current state of the network to a given pattern or state.

        :param pattern: A NumPy array representing a pattern/state which the network is set to.
        """

        self.state = np.copy(pattern)

    def update_step(self, batch=True, starting_step=None, step_callback=None, bias=0.0, output_bias=0.0,
                    output_scaling=1.0):
        """
        Calculates the next state of a Hopfield network using the current state and the weight matrix.

        :param batch: Optional. Determines if the update step is done synchronously (batch) or asynchronously (sequential). True (synchronous) by default.
        :param starting_step: Optional. Starting step to be used in step_callback function. Useless if step_callback not defined.
        :param step_callback: Callback function for sequential (batch=False) learning. Arguments given are a copy of the current state and the number of update starting with 1.
        :param bias: Optional. Adds negative bias to updating term.
        :param output_bias: Optional. Adds a bias to all outputs.
        :param output_scaling: Optional. Multiplies output. Multiplication happens before adding output bias.
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
            self.state = np.matmul(self.state, self.weights) - bias
            self.state = np.where(self.state >= 0, 1, -1)  # sign function across new state
            self.state = output_bias + output_scaling * self.state
        # asynchronous update
        else:
            random_sequence = np.array([i for i in range(self.number_of_neurons)])
            np.random.shuffle(random_sequence)
            for step, neuron_idx in enumerate(random_sequence, 1):
                # calculate new state and use the sign function on the result
                self.state[neuron_idx] = 1 if (np.matmul(self.state, self.weights.T[neuron_idx]) - bias) >= 0 else -1
                self.state[neuron_idx] = output_bias + output_scaling * self.state[neuron_idx]
                if step_callback:
                    step_callback(np.copy(self.state), starting_step + step)

        return np.copy(self.state)
