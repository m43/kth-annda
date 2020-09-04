import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from assignment_2.util.time_series import generate_time_series_transposed

# constants
FOLDER_PATH = 'results/'

# inputs and outputs
inputs, outputs = generate_time_series_transposed(1.5)
print('No. of inputs:', len(inputs[0]))
print('No. of outputs:', len(outputs))

plt.plot(range(300, 1500), outputs)
plt.savefig(fname=FOLDER_PATH + 'series')
plt.show()

# training data
x_train = inputs[0:800]
y_train = outputs[0:800]

x_train = tf.convert_to_tensor(x_train, dtype='float64')
y_train = tf.convert_to_tensor(y_train, dtype='float64')

print('No. of training inputs:', len(x_train))
print(x_train)
print('No. of training outputs:', len(y_train))
print(y_train)

# validation data
x_val = inputs[800:1000]
y_val = outputs[800:1000]

x_val = tf.convert_to_tensor(x_val, dtype='float64')
y_val = tf.convert_to_tensor(y_val, dtype='float64')

print('No. of validation inputs:', len(x_val))
print(x_val)
print('No. of validation outputs:', len(y_val))
print(y_val)

# test data
x_test = inputs[1000:1201]
y_test = outputs[1000:1201]

x_test = tf.convert_to_tensor(x_test, dtype='float64')
y_test = tf.convert_to_tensor(y_test, dtype='float64')

print('No. of test inputs:', len(x_test))
print(x_test)
print('No. of test outputs:', len(y_test))
print(y_test)

# efficient pipeline input (dataset)
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(32)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(Model):

    def get_config(self):
        pass

    # TODO: add regularization to layers
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(8, input_shape=[5, ], activation=tf.nn.relu, dtype='float64', use_bias=True,
                        activity_regularizer=tf.keras.regularizers.l1(0.1))  # hidden 1
        # self.d2 = Dense(4, activation=tf.nn.relu, dtype='float64', use_bias=True,
        #                 activity_regularizer=tf.keras.regularizers.l1(0.1))  # hidden 2
        self.d3 = Dense(1, dtype='float64', use_bias=True,
                        activity_regularizer=tf.keras.regularizers.l1(0.1))  # output

    def call(self, x, **kwargs):
        x = self.d1(x)
        # x = self.d2(x)
        return self.d3(x)


# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

loss = tf.keras.metrics.MeanSquaredError(name='loss')
mae = tf.keras.metrics.MeanAbsoluteError(name='mae')


@tf.function
def train_step(patterns, targets):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(patterns, training=True)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # test_loss(t_loss)
    train_loss(targets, predictions)
    train_mae(targets, predictions)


@tf.function
def test_step(patterns, targets):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(patterns, training=False)
    # t_loss = loss_object(targets, predictions)

    # test_loss(t_loss)
    loss(targets, predictions)
    mae(targets, predictions)


EPOCHS = 10000
EARLY_STOP = 100
pocket_loss = float('inf')
pocket_epoch = 0
last_epoch = 0
train_loss_log = []
val_loss_log = []
loss_unchanged_counter = 0

for epoch in range(EPOCHS):
    last_epoch = epoch
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_mae.reset_states()
    loss.reset_states()
    mae.reset_states()

    for step, (patterns, targets) in enumerate(train_ds):
        train_step(patterns, targets)

    for val_patterns, val_targets in val_ds:
        test_step(val_patterns, val_targets)

    if loss.result() < pocket_loss:
        loss_unchanged_counter = 0
        model.save_weights(filepath=FOLDER_PATH + 'temp/pocket_weights')
        pocket_loss = loss.result()
        pocket_epoch = epoch
    else:
        loss_unchanged_counter += 1
        if loss_unchanged_counter > 100:
            break

    template = 'Epoch {}, Loss (MSE): {}, MAE: {}, Validation Loss (MSE): {}, Validation MAE: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_mae.result(),
                          loss.result(),
                          mae.result()))

    train_loss_log.append(train_loss.result())
    val_loss_log.append(loss.result())

    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_mae.reset_states()
    loss.reset_states()
    mae.reset_states()

plt.plot(range(0, last_epoch), train_loss_log[0:last_epoch], label='train MSE')
plt.plot(range(0, last_epoch), val_loss_log[0:last_epoch], label='validation MSE')
plt.scatter(pocket_epoch, pocket_loss, c='#FF0000', label='point of lowest validation MSE')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()
plt.savefig(fname=FOLDER_PATH + 'learning_mse')

# restore best weights
model.load_weights(filepath=FOLDER_PATH + 'temp/pocket_weights')

for test_patterns, test_targets in test_ds:
    test_step(test_patterns, test_targets)

template = 'Final evaluation (restored best weights) --\n Test Loss (MSE): {} Test MAE: {}'
print(template.format(loss.result(),
                      mae.result()))

print('Final weights:', model.d1.get_weights())

# TODO: a histogram that makes sense
plt.hist(model.d1.get_weights())
plt.ylabel('values')
plt.legend()
plt.show()
plt.savefig(fname=FOLDER_PATH + 'weight_histogram')

train_outputs = model(x_train)

val_outputs = None
test_outputs = None

for test_patterns, test_targets in val_ds:
    if val_outputs is None:
        val_outputs = model(test_patterns)
    else:
        val_outputs = tf.concat((val_outputs, model(test_patterns)), 0)

for test_patterns, test_targets in test_ds:
    if test_outputs is None:
        test_outputs = model(test_patterns)
    else:
        test_outputs = tf.concat((test_outputs, model(test_patterns)), 0)

plt.plot(range(300, 1500), outputs, c='#0000FF', label='time series')
plt.plot(range(300, 1100), train_outputs, linestyle='--', c='#FF7F00',
         label='training predictions')  # sort for batch
plt.plot(range(1100, 1300), val_outputs, linestyle='--', c='#FF3F00', label='validation predictions')
plt.plot(range(1300, 1500), test_outputs, linestyle='--', c='#FF0000', label='test predictions')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.savefig(fname=FOLDER_PATH + 'series_and_test_predictions')
plt.show()
