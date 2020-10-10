from demo.lab4.rbm import RestrictedBoltzmannMachine
from demo.lab4.util import *

MNIST_LOCATION = "datasets/mnist"

if __name__ == "__main__":
    np.random.seed(36)
    ndim_hidden = 500
    epochs = 50
    batch_size = 10
    # weight_decay = 1e-5
    weight_decay = 0
    learning_rate = 0.01
    momentum = 0.7
    imgs_folder = f"imgs/lab4"
    run_name = f"rbm_14" \
               f"_nh={ndim_hidden}" \
               f"_lr={learning_rate}" \
               f"_wd={weight_decay:1.1e}" \
               f"_m={momentum}" \
               f"_b={batch_size}" \
               f"x_e={epochs}".replace(".", ",")  # f"_nh={ndim_hidden}" \
    debug_save_folder = os.path.join(imgs_folder, run_name)

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(MNIST_LOCATION, dim=image_size, n_train=60000,
                                                              n_test=10000)
    ''' restricted boltzmann machine '''

    print("\nStarting a Restricted Boltzmann Machine..")

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                     ndim_hidden=ndim_hidden,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=batch_size,
                                     visuals_save_path=debug_save_folder,
                                     )

    rbm.cd1(visible_trainset=train_imgs, n_iterations=epochs * train_imgs.shape[0] // batch_size)
    rbm.viz_all_rf()
