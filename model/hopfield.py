import numpy as np
import sys


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
        self.weights = None
        self.state = None

    def learn_patterns(self, patterns, scaling=True, self_connections=False):
        """
        Sets weights of a Hopfield network using the Hebbian one-shot rule using the given patterns. Will delete previous weights if there were any.

        :param patterns: A NumPy matrix of patterns to be learned. Each row represents a pattern. The length of each row (the number of columns) must be equal to the number of neurons in the network.
        :param scaling: Optional. Determines if weights will be scaled with the reciprocal value of the number of patterns or not. True by default.
        :param self_connections: Optional. Determines if neurons are connected to themselves or not. False by default.
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

    def update_automatically(self, batch=True, update_cap=100000, sequential_stability_cap=100, step_callback=None):
        """
        Calculates states using the update rule until convergence, oscillation (only when using synchronous updates) or reaching a defined cap.

        :param batch: Optional. Determines if updating is done synchronously (batch) or asynchronously (sequential). True (synchronous) by default.
        :param update_cap: Optional. Determines how many updates will happen are done without convergence. If the cap is reached updating stops and the function returns the last and current state. 10000 by default.
        :param sequential_stability_cap: Optional. Determines how many times the update step has to return the same step for it to be stable. 10 by default.
        :param step_callback: Optional. A callback function called after each step update. Arguments given are a copy of the current state and the number of update starting with 1.
        :return: Current state of the Hopfield network after doing one full update.
        """
        if self.debug:
            print('\t-----------------------------')
            print(f'\tStarting automatic updating process.\n'
                  f'\tStarting state = {self.state}\n'
                  f'\tUpdating is capped at {update_cap} steps.')
            if batch:
                print('\tUsing synchronous (batch) updating.')
            else:
                print(f'\tUsing asynchronous (sequentital) updating.\n'
                      f'\tLearning stops after reaching the same state {sequential_stability_cap} times.')
            print('\t-----------------------------')

        step = 0
        sequential_stable_counter = 0
        previous_states = set()
        previous_states.add(tuple(np.copy(self.state)))
        last_state = np.copy(self.state)
        while step <= update_cap:

            self.update_step(batch=batch)
            step += 1

            if step_callback is not None:
                step_callback(np.copy(self.state), step)

            # check convergence
            if (last_state == self.state).all():
                if batch:
                    if self.debug:
                        print(f'\tStable point reached after {step} steps, stopping learning process.')
                    break
                elif not batch and sequential_stable_counter >= sequential_stability_cap:
                    if self.debug:
                        print(
                            f'\tStable point reached after {step} steps (sequential learning had the same state for {sequential_stability_cap} updates), stopping learning process.')
                    break
                else:
                    sequential_stable_counter += 1
            # check oscillation
            elif batch and tuple(self.state) in previous_states:
                if self.debug:
                    print('\tOscillation appeared while synchronously updating state, stopping learning process.')
                break
            else:
                last_state = np.copy(self.state)
                sequential_stable_counter = 0
                if batch:
                    previous_states.add(tuple(np.copy(self.state)))

        return np.copy(self.state)

    def set_state(self, pattern):
        """
        Sets the current state of the network to a given pattern or state.

        :param pattern: A NumPy array representing a pattern/state which the network is set to.
        """

        self.state = np.copy(pattern)

    def update_step(self, batch=True):
        """
        Calculates the next state of a Hopfield network using the current state and the weight matrix.

        :param batch: Optional. Determines if the update step is done synchronously (batch) or asynchronously (sequential). True (synchronous) by default.
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
            self.state = np.where(self.state >= 0, 1, -1)  # sign function across new state
        # asynchronous update
        else:
            random_sequence = np.array([i for i in range(self.number_of_neurons)])
            np.random.shuffle(random_sequence)
            for neuron_idx in random_sequence:
                # calculate new state and use the sign function on the result
                self.state[neuron_idx] = 1 if np.matmul(self.state, self.weights.T[neuron_idx]) >= 0 else -1

        return np.copy(self.state)
