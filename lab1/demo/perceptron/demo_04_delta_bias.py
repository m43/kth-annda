from demo.perceptron.util import delta_rule_learning_demo
from demo.util import perpare_reproducable_separable_dataset, \
    perpare_reproducable_separable_dataset_impossible_with_no_bias
from utils.util import ensure_dir, TwoClassDatasetGenerator

if __name__ == '__main__':
    ##################################################
    #### Delta rule - separable data and no bias #####
    ##################################################

    save_folder = "../../imgs"
    eta = 0.001  # learning rate
    batch_size = 200
    max_iter = 10000  # max number of epochs
    debug = True
    delta_n = 50  # number of epochs without improvements in delta learning
    delta_n_batch = 150
    bias = False

    # This dataset can be separated without bias
    inputs, targets = perpare_reproducable_separable_dataset()

    # This dataset CANNOT be separated without bias
    inputs_impossible, targets_impossible = perpare_reproducable_separable_dataset_impossible_with_no_bias()

    ensure_dir(save_folder)
    delta_rule_learning_demo(inputs, targets, f"DELTA_BATCH_NO_BIAS_1", debug, save_folder, max_iter, eta,
                             (delta_n if batch_size == 1 else delta_n_batch), batch_size, bias)
    delta_rule_learning_demo(inputs_impossible, targets_impossible, f"DELTA_BATCH_NO_BIAS_2", debug, save_folder,
                             max_iter, eta, (delta_n if batch_size == 1 else delta_n_batch), batch_size, bias)
