import matplotlib.pyplot as plt
import numpy as np

from model.slp import SingleLayerPerceptron
from utils.util import shuffle_two_arrays, ensure_dir

TEST = True
PLOT_FOLDER = 'imgs'
ETA = 0.0001  # learning rate
MAX_ITER = 1000  # max number of epochs
DELTA_N = 100  # number of epochs without improvements in delta batch learning


def plot_all(first_class, second_class, line_coefficients, name):
    ensure_dir(PLOT_FOLDER)
    plt.title(name)
    plt.scatter(first_class[0], first_class[1], c='#FF0000')
    plt.scatter(second_class[0], second_class[1], c='#00FF00')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(plt.xlim())  # lock limit on the x axis
    plt.ylim(plt.ylim())  # lock limit on the y axis
    if line_coefficients is not None:
        x = np.linspace(-5, 5, 100)
        y = (- line_coefficients[0, 0] * x - line_coefficients[0, 2]) / line_coefficients[0, 1]
        plt.plot(x, y)
    plt.savefig(PLOT_FOLDER + '/' + name)
    plt.show()


def test_print(*a, **b):
    if not TEST:
        return
    print(*a, **b)


if __name__ == '__main__':
    np.random.seed(32)

    # define number of samples per class, means for each sample dimension and std
    # 3 sigma rule; probability of linear inseparability insignificant
    n = 100
    m_a = [1.5, 1.5]
    sigma_a = 0.5
    m_b = [-1.5, -1.5]
    sigma_b = 0.5

    # create class A data with bias and labels (1)
    class_a = np.array([np.random.normal(m_a[0], sigma_a, n),
                        np.random.normal(m_a[1], sigma_a, n)])
    test_print('Class A pattern:\n', class_a, '\n')

    # create class B data with labels (-1)
    class_b = np.array([np.random.normal(m_b[0], sigma_b, n),
                        np.random.normal(m_b[1], sigma_b, n)])
    test_print('Class B pattern:\n', class_b, '\n')

    # targets
    targets = np.concatenate((np.ones(n), np.ones(n) * -1), axis=0).reshape(1, -1)

    # join and shuffle class A and class B together (shuffling important for sequential learning)
    patterns = np.concatenate((class_a, class_b), axis=1)  # column-wise concatenation
    patterns, targets = shuffle_two_arrays(patterns, targets)
    test_print('All patterns (shuffled):\n', patterns, '\n')
    test_print('All targets (shuffled):\n', targets, '\n')
    plot_all(class_a, class_b, line_coefficients=None, name='samples')

    print("PERCEPTRON LEARNING RULE ")
    slp = SingleLayerPerceptron(patterns, targets, debug=TEST)
    print("Before training")
    test_print(slp.W)
    slp.eval(patterns, targets, delta_metrics=False)
    plot_all(class_a, class_b, line_coefficients=slp.W, name='Perceptron learning - start')

    slp.train(patterns, targets, ETA, minibatch_size=1, max_iter=MAX_ITER, delta=False)

    print("After training")
    test_print(slp.W)
    slp.eval(patterns, targets, delta_metrics=False)
    plot_all(class_a, class_b, line_coefficients=slp.W, name='Perceptron learning - end result')

    print("DELTA RULE PERCEPTRON")
    slp = SingleLayerPerceptron(patterns, targets, debug=TEST)

    print("Before training")
    slp.eval(patterns, targets)
    plot_all(class_a, class_b, line_coefficients=slp.W, name='Delta rule learning - start')

    slp.train(patterns, targets, ETA, minibatch_size=1, max_iter=MAX_ITER)

    print("After training")
    slp.eval(patterns, targets)
    plot_all(class_a, class_b, line_coefficients=slp.W, name='Delta rule learning - end result')

    print("DELTA RULE PERCEPTRON BATCH")
    slp = SingleLayerPerceptron(patterns, targets, debug=TEST)

    print("Before training")
    slp.eval(patterns, targets)
    plot_all(class_a, class_b, line_coefficients=slp.W, name='Delta rule learning BATCH - start')

    slp.train(patterns, targets, ETA, minibatch_size=targets.shape[-1], max_iter=MAX_ITER)

    print("After training")
    slp.eval(patterns, targets)
    plot_all(class_a, class_b, line_coefficients=slp.W, name='Delta rule learning BATCH - end result')

# TODO refactor file. Create separate demo programs
