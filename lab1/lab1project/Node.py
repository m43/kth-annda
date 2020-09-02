import numpy


class Node:

    def __init__(self, weights):
        self.weights = weights

    def calculate_output(self, input):
        return self.weights * input

    def update_weights(self, delta):
        self.weights *= delta
