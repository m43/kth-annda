from dbn import DeepBeliefNet
from rbm import RestrictedBoltzmannMachine
from util import *
from matplotlib import pyplot as plt


MNIST_LOCATION = "../../datasets/mnist"

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(MNIST_LOCATION, dim=image_size, n_train=60000,
                                                              n_test=10000)

    ''' restricted boltzmann machine '''

    print("\nStarting a Restricted Boltzmann Machine, 500 hidden nodes")

    rbm500 = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
                                    )
    print("\nTraining..")
    rl500 = rbm500.cd_epochs(visible_trainset=train_imgs, n_epochs=25)


    plt.plot(list(range(len(rl500))), rl500, "r-", label="500")

    print("\nStarting a Restricted Boltzmann Machine, 200 hidden nodes")

    rbm200 = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                     ndim_hidden=200,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
                                    )
    print("\nTraining..")

    rl200 = rbm200.cd_epochs(visible_trainset=train_imgs, n_epochs=25)
    plt.plot(list(range(len(rl200))), rl200, "b-", label="200")
    plt.xlabel("iteration")
    plt.ylabel("recon loss")

    plt.title("RBM: Losses in respect to number of hidden node")
    plt.legend()
    plt.savefig(fname=f'RMB_recon', bbox_inches='tight')
    plt.close()
