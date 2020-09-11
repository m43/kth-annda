import shutil
import statistics
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import sys

from assignment_2.model.model import TensorFlowModel, train_step_wrapper, evaluate_step_wrapper
from assignment_2.util.time_series import generate_time_series
from assignment_2.util.utils import generate_in_out_tensors

# length of training - parameters
NUMBER_OF_TESTS = 10  # number of models to be tested before results are compiled ( > 0)
MAX_EPOCHS = 20000  # number of epochs before learning stops
EARLY_STOP = 50  # number of epochs before stopping training if no improvement in the validation set is visible
TOLERANCE = 1e-5  # amount of improvement needed

# model - parameters
LEARNING_RATE = 0.15
REGULARIZATION_METHOD = 'l2'  # l1 or l2
REGULARIZATION_RATE = 0.1
NODES = [8]  # list of numbers of nodes in each hidden layer
NOISE = True
NOISE_SIGMA = 0.09
# read arguments if there are any
if sys.argv[1:]:
    # if noise is used as an argument add a second layer instead of using the value for the first one
    if sys.argv[3:]:
        NODES.append(int(sys.argv[1]))
    else:
        NODES = [int(sys.argv[1])]
if sys.argv[2:]:
    REGULARIZATION_RATE = float(sys.argv[2])
if sys.argv[3:]:
    NOISE = True
    NOISE_SIGMA = float(sys.argv[3])

SHOW_PLOTS = False
DEBUG = False  # if True output is verbose
SAVE_DIRECTORY = 'results/'  # folder in which figures will be saved
CONFIG_DIRECTORY = 'lr={}_{}-reg={}_hidden-shape={}_tests={}_early_stop={}_tolerance={}'.format(LEARNING_RATE,
                                                                                                REGULARIZATION_METHOD,
                                                                                                REGULARIZATION_RATE,
                                                                                                NODES, NUMBER_OF_TESTS,
                                                                                                EARLY_STOP, TOLERANCE)
if NOISE:
    CONFIG_DIRECTORY += '_noise={}'.format(NOISE_SIGMA)
CONFIG_DIRECTORY += '/'

PLOT_DPI = 300  # DPI of figures for saving
GLOBAL_COUNTER = 0

# tensorflow additional logging
# tf.debugging.set_log_device_placement(True)

# directory creation
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
if not os.path.exists(SAVE_DIRECTORY + CONFIG_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY + CONFIG_DIRECTORY)
if not os.path.exists(SAVE_DIRECTORY + 'temp'):
    os.makedirs(SAVE_DIRECTORY + 'temp')

# set default type to float64
tf.keras.backend.set_floatx('float64')


def run_model(index):
    # inputs and outputs
    inputs, outputs = generate_time_series(1.5, noise=NOISE, noise_sigma=NOISE_SIGMA)

    if DEBUG:
        print('No. of inputs:', len(inputs[0]))
        print('No. of outputs:', len(outputs))

    if not NOISE:
        save_folder = SAVE_DIRECTORY + 'series'
    else:
        save_folder = SAVE_DIRECTORY + CONFIG_DIRECTORY + 'noisy_series_' + str(index)

    # plot Mackey-Glass time series
    plt.plot(range(300, 1500), outputs, label='Mackey-Glass time series')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()
    plt.savefig(save_folder, dpi=PLOT_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # training data
    x_train, y_train = generate_in_out_tensors(inputs, outputs, start=0, end=800, verbose=DEBUG)

    # validation data
    x_val, y_val = generate_in_out_tensors(inputs, outputs, start=800, end=1000, verbose=DEBUG)

    # test data
    x_test, y_test = generate_in_out_tensors(inputs, outputs, start=1000, end=1201, verbose=DEBUG)

    # efficient pipeline inputs (dataset)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(800)
    # add .shuffle(buffer_size=1024, reshuffle_each_iteration=True) before batch for shuffling
    # (not needed if using whole set)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # create an instance of the model
    model = TensorFlowModel(learning_rate=LEARNING_RATE, nodes=NODES,
                            regularization_method=REGULARIZATION_METHOD, regularization_rate=REGULARIZATION_RATE)

    # loss and extra metric to be evaluated during training
    train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
    train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

    # loss and extra metric to be evaluated during validation (and testing)
    loss = tf.keras.metrics.MeanSquaredError(name='loss')
    mae = tf.keras.metrics.MeanAbsoluteError(name='mae')

    pocket_loss = float('inf')
    pocket_epoch = 0
    last_epoch = 0
    train_loss_log = []
    val_loss_log = []
    loss_unchanged_counter = 0

    # get tf.functions for training and evaluation
    train_step = train_step_wrapper()
    evaluate_step = evaluate_step_wrapper()

    for epoch in range(MAX_EPOCHS + 1):
        last_epoch = epoch
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_mae.reset_states()
        loss.reset_states()
        mae.reset_states()

        for patterns, targets in train_ds:
            train_step(model, patterns, targets, train_loss, train_mae)

        for val_patterns, val_targets in val_ds:
            evaluate_step(model, val_patterns, val_targets, loss, mae)

        if loss.result() + TOLERANCE < pocket_loss:
            loss_unchanged_counter = 0

            # prevents errors with renaming (.save_weights method)
            if os.path.exists(SAVE_DIRECTORY + 'temp'):
                shutil.rmtree(SAVE_DIRECTORY + 'temp')
                os.mkdir(SAVE_DIRECTORY + 'temp')

            model.save_weights(filepath=SAVE_DIRECTORY + 'temp/pocket_weights')

            pocket_loss = loss.result()
            pocket_epoch = epoch
        else:
            loss_unchanged_counter += 1
            if loss_unchanged_counter > EARLY_STOP:
                if not DEBUG:
                    if epoch % 100 == 0:
                        print('=', end='')
                        if epoch >= MAX_EPOCHS:
                            print(']')
                    else:
                        print('/', end='')
                    print(']; training took', epoch, 'epochs.')
                break

        if DEBUG:
            template = 'Epoch {}, Loss (MSE): {}, MAE: {}, Validation Loss (MSE): {}, Validation MAE: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_mae.result(),
                                  loss.result(),
                                  mae.result()))
        else:
            if epoch == 0:
                print('Training a model: [', end='')
            elif epoch % 100 == 0:
                print('=', end='')
                if epoch >= MAX_EPOCHS:
                    print(']')

        train_loss_log.append(train_loss.result())
        val_loss_log.append(loss.result())

    # plot learning process
    plt.plot(range(0, last_epoch), train_loss_log[0:last_epoch], label='train MSE')
    plt.plot(range(0, last_epoch), val_loss_log[0:last_epoch], label='validation MSE')
    plt.scatter(pocket_epoch, pocket_loss, c='#FF0000', label='point of lowest validation MSE')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.ylim(top=0.05, bottom=0.0)
    plt.legend()
    plt.savefig(SAVE_DIRECTORY + CONFIG_DIRECTORY + 'learning_mse_' + str(index), dpi=PLOT_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # reset metrics for final evaluation
    loss.reset_states()
    mae.reset_states()

    # restore best weights
    model.load_weights(filepath=SAVE_DIRECTORY + 'temp/pocket_weights')

    for test_patterns, test_targets in test_ds:
        evaluate_step(model, test_patterns, test_targets, loss, mae)

    template = 'Final evaluation (best weights) --\n Test Loss (MSE): {} Test MAE: {}'
    print(template.format(loss.result(),
                          mae.result()))

    test_loss = float(loss.result())

    if DEBUG:
        print('Final weights:')
        for layer in model.layer_list:
            print(layer, layer.get_weights())

    # histogram of weights
    all_weights = []
    for weights in model.layer_list[0].get_weights()[0]:
        for weight in weights:
            all_weights.append(float(weight))
    plt.hist(all_weights, label='weights')
    plt.ylabel('values')
    plt.legend()
    plt.savefig(SAVE_DIRECTORY + CONFIG_DIRECTORY + 'weight_histogram_' + str(index), dpi=PLOT_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # get outputs and plot them against real data
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

    plt.plot(range(300, 1500), outputs, c='#0000FF', label='time series', linewidth=2.5)
    plt.plot(range(300, 1100), train_outputs, c='#FFA500', label='training predictions')
    plt.plot(range(1100, 1300), val_outputs, c='#00CC00', label='validation predictions')
    plt.plot(range(1300, 1500), test_outputs, c='#F70000', label='test predictions')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()
    plt.savefig(SAVE_DIRECTORY + CONFIG_DIRECTORY + 'series_and_test_predictions_' + str(index), dpi=PLOT_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # get last train error
    loss.reset_states()
    mae.reset_states()

    for train_patterns, train_targets in train_ds:
        evaluate_step(model, train_patterns, train_targets, loss, mae)

    train_loss = float(loss.result())

    # get last validation error
    loss.reset_states()
    mae.reset_states()

    for val_patterns, val_targets in val_ds:
        evaluate_step(model, val_patterns, val_targets, loss, mae)

    val_loss = float(loss.result())

    return test_loss, float(pocket_epoch), model, train_loss, val_loss


def main():
    print('Running', NUMBER_OF_TESTS, 'test with LR=', LEARNING_RATE, ', regularization', REGULARIZATION_METHOD,
          'with strength of', REGULARIZATION_RATE, ', shape of networks is:', NODES)

    if NOISE:
        print('Using noise with sigma =', NOISE_SIGMA)

    start = time.time()
    loss = []
    train_loss = []
    val_loss = []
    epoch = []
    models = []
    best_model = None
    best_loss = float('inf')
    best_epoch = 0
    for i in range(NUMBER_OF_TESTS):
        temp_loss, temp_epoch, temp_model, temp_train_loss, temp_val_loss = run_model(i)
        if temp_loss < best_loss:
            best_model = temp_model
            best_loss = temp_loss
            best_epoch = temp_epoch
        loss.append(float(temp_loss))
        train_loss.append(temp_train_loss)
        val_loss.append(temp_val_loss)
        epoch.append(temp_epoch)
        models.append(temp_model)

    log_entry = 'Finished evaluating {} models.\n' \
                'Learning rate = {}\n' \
                'Number of hidden layers = {}\n' \
                'Hidden layers\' nodes: {}\n' \
                'Regularization method: {}\n' \
                'Regularization rate of: {}\n' \
                'Number of tests: {}\n'.format(NUMBER_OF_TESTS, LEARNING_RATE, len(NODES), NODES,
                                               REGULARIZATION_RATE, REGULARIZATION_RATE, NUMBER_OF_TESTS)

    with open(SAVE_DIRECTORY + CONFIG_DIRECTORY + 'log.txt', 'a') as log:
        log.write(log_entry)

    print(log_entry)

    if NUMBER_OF_TESTS > 1:
        loss_mean = statistics.mean(loss)
        loss_std = statistics.stdev(loss)

        print('Loss mean:', loss_mean)
        print('Loss std:', loss_std)

        val_loss_mean = statistics.mean(val_loss)
        train_loss_mean = statistics.mean(train_loss)

        epoch_mean = statistics.mean(epoch)
        epoch_std = statistics.stdev(epoch)

        print('Epoch mean:', epoch_mean)
        print('Epoch std:', epoch_std)

        with open(SAVE_DIRECTORY + CONFIG_DIRECTORY + 'log.txt', 'a') as log:
            log.write('Losses: {}\n'
                      'Loss mean: {}\n'
                      'Loss std: {}\n'
                      'Training loss mean: {}\n'
                      'Validation loss mean: {}\n'
                      'Epochs: {}\n'
                      'Epoch mean: {}\n'
                      'Epoch std: {}\n'
                      'Best model had an MSE of: {}, and stopped training after {} epochs.\n'
                      .format(loss, loss_mean, loss_std, train_loss_mean, val_loss_mean, epoch, epoch_mean, epoch_std,
                              float(best_loss), best_epoch))

    else:
        with open(SAVE_DIRECTORY + CONFIG_DIRECTORY + 'log.txt', 'a') as log:
            log.write('Model had an MSE of {} on the test set\n'
                      'The train error was {}, and the validation error {}\n'.format(float(best_loss), train_loss[0],
                                                                                     val_loss[0]))

    # plot accumulated weights
    all_weights = []
    for model in models:
        for weights in model.layer_list[0].get_weights()[0]:
            for weight in weights:
                all_weights.append(weight)

    plt.hist(all_weights)
    plt.xlim(left=-2, right=2)
    plt.ylim(bottom=0, top=200)
    plt.xlabel('weight value')
    plt.ylabel('count')
    plt.savefig(fname=SAVE_DIRECTORY + CONFIG_DIRECTORY + 'weight_histogram')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    end = time.time()
    with open(SAVE_DIRECTORY + CONFIG_DIRECTORY + 'log.txt', 'a') as log:
        log.write('The above took: {}\n'.format(end - start))

    # remove temp at the end
    if os.path.exists(SAVE_DIRECTORY + '/temp'):
        shutil.rmtree(SAVE_DIRECTORY + '/temp')


if __name__ == '__main__':
    main()
