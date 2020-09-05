from demo.perceptron.demo_util import delta_rule_learning_demo, \
    perpare_reproducable_separable_dataset
from utils.util import ensure_dir

if __name__ == '__main__':
    ########################
    #### CONFIGURATION #####
    ########################
    save_folder = "../../imgs"
    eta = 0.002  # learning rate
    batch_size = 200
    max_iter = 10000  # max number of epochs
    debug = True
    delta_n = 50  # number of epochs without improvements in delta learning
    delta_n_batch = 150
    bias = True
    inputs, targets = perpare_reproducable_separable_dataset()

    ##########################
    #### Single run DEMO #####
    ##########################
    ensure_dir(save_folder)
    delta_rule_learning_demo(
        inputs, targets, f"test",
        debug, save_folder, max_iter, eta, (delta_n if batch_size == 1 else delta_n_batch), batch_size, bias)
