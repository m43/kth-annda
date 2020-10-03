# from demo.lab4.dbn import DeepBeliefNet
# from demo.lab4.rbm import RestrictedBoltzmannMachine
# from demo.lab4.util import *

from dbn import DeepBeliefNet
from rbm import RestrictedBoltzmannMachine
from util import *

MNIST_LOCATION = "../../datasets/mnist"

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(MNIST_LOCATION, dim=image_size, n_train=60000,
                                                              n_test=10000)
    # train_imgs, train_lbls = train_imgs[:123], train_lbls[:123] # speed debuging

    # ''' restricted boltzmann machine '''

    # print("\nStarting a Restricted Boltzmann Machine..")

    # rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
    #                                  ndim_hidden=200,
    #                                  is_bottom=True,
    #                                  image_size=image_size,
    #                                  is_top=False,
    #                                  n_labels=10,
    #                                  batch_size=20
    #                                  )

    # rbm.cd1(visible_trainset=train_imgs, n_iterations=10000)
    # rbm.cd_epochs(visible_trainset=train_imgs, n_epochs=10)
    ''' deep- belief net '''

    print("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis": image_size[0] * image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
                        )

    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=15)

    # dbn.recognize(train_imgs, train_lbls)

    # dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    # ''' fine-tune wake-sleep training '''

    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=2000)

    # dbn.recognize(train_imgs, train_lbls)

    # dbn.recognize(test_imgs, test_lbls)

    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1, 10))
    #     digit_1hot[0, digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")