from demo.lab4.dbn import DeepBeliefNet
from demo.lab4.util import *

MNIST_LOCATION = "datasets/mnist"

if __name__ == "__main__":
    np.random.seed(36)
    epochs = 50
    batch_size = 10
    weight_decay = 1e-5
    # weight_decay = 0
    learning_rate = 0.01
    momentum = 0.7
    imgs_folder = f"imgs/lab4"
    run_name = f"dbn_15" \
               f"_lr={learning_rate}" \
               f"_wd={weight_decay:1.1e}" \
               f"_m={momentum}" \
               f"_b={batch_size}" \
               f"x_e={epochs}".replace(".", ",")
    debug_save_folder = os.path.join(imgs_folder, run_name)

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(MNIST_LOCATION, dim=image_size, n_train=60000,
                                                              n_test=10000)

    ''' deep- belief net '''

    print("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis": image_size[0] * image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10,
                        save_path=debug_save_folder,
                        rbm_weight_decay=weight_decay,
                        rbm_momentum=momentum,
                        rbm_learning_rate=learning_rate
                        )

    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls,
                              n_iterations=epochs * train_imgs.shape[0] // batch_size)

    dbn.recognize(train_imgs, train_lbls, debug=True)
    dbn.recognize(test_imgs, test_lbls, debug=True)
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="rbms")
        print(f"Generated {digit}")
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="rbms_2_")
        print(f"Generated {digit}")

    # ''' fine-tune wake-sleep training '''
    #
    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=2000)
    #
    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)
    #
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1, 10))
    #     digit_1hot[0, digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
