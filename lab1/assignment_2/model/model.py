import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense


class TensorFlowModel(Model):

    def get_config(self):
        pass

    def __init__(self, learning_rate, hidden_layers, nodes, regularization_method, regularization_rate):
        super(TensorFlowModel, self).__init__()

        if hidden_layers < 1:
            raise RuntimeError('Wrong argument hidden_layers: models must have at least one hidden layer')
        elif hidden_layers != len(nodes):
            raise RuntimeError(
                'Wrong argument nodes: nodes must be an array type with as many elements as there are hidden layers')

        if regularization_method == 'l1':
            regularizer = tf.keras.regularizers.l1(regularization_rate)
        elif regularization_method == 'l2':
            regularizer = tf.keras.regularizers.l2(regularization_rate)
        else:
            raise RuntimeError('No known regularizer defined, regularization method most be l1 or l2')

        self.optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
        self.loss_object = tf.losses.MeanSquaredError()

        # first hidden layer (needs to define input)
        self.layer_list = [Dense(nodes[0], input_shape=[5, ], activation=tf.nn.sigmoid, dtype='float64', use_bias=True,
                                 activity_regularizer=regularizer)]

        # other hidden layers
        for i in range(1, hidden_layers):
            self.layer_list.append(Dense(nodes[i], activation=tf.nn.sigmoid, dtype='float64', use_bias=True,
                                         activity_regularizer=regularizer))

        # output layer
        self.layer_list.append(Dense(1, dtype='float64', use_bias=True,
                                     activity_regularizer=regularizer))

    def call(self, x, **kwargs):
        for layer in self.layer_list:
            x = layer(x)
        return x


# wrapper is defined in order to be able to run more models
# each model which calls a wrapper receives a different instance of the tf.function
# (avoids ValueError: tf.function-decorated function tried to create variables on non-first call)
def train_step_wrapper():
    @tf.function
    def train_step(model, patterns, targets, loss, extra_metric):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(patterns, training=True)
            calculated_loss = model.loss_object(targets, predictions)
        gradients = tape.gradient(calculated_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss(targets, predictions)
        extra_metric(targets, predictions)

    return train_step


# wrapper is defined in order to be able to run more models
# each model which calls a wrapper receives a different instance of the tf.function
# (avoids ValueError: tf.function-decorated function tried to create variables on non-first call)
def evaluate_step_wrapper():
    @tf.function
    def evaluate_step(model, patterns, targets, loss, extra_metric):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(patterns, training=False)

        loss(targets, predictions)
        extra_metric(targets, predictions)

    return evaluate_step
