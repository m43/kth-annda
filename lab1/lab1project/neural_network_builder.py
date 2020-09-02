import numpy as np
import matplotlib.pyplot as plt

TEST = True
PLOT_FOLDER = 'results'
ETA = 0.001  # learning rate
N = 100000  # number of epochs
DELTA_N = 100  # number of epochs without improvements in delta batch learning


def test_print(*a, **b):
    if not TEST:
        return
    print(*a, **b)


def plot_all(first_class, second_class, line_coefficients, name):
    plt.scatter(first_class[0], first_class[1], c='#FF0000')
    plt.scatter(second_class[0], second_class[1], c='#00FF00')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(plt.xlim())  # lock limit on the x axis
    plt.ylim(plt.ylim())  # lock limit on the y axis
    if line_coefficients is not None:
        x = np.linspace(-5, 5, 100)
        y = (- line_coefficients[0] * x - line_coefficients[2]) / line_coefficients[1]
        plt.plot(x, y)
    plt.savefig(PLOT_FOLDER + '/' + name)
    plt.show()


def shuffle_two_arrays(a, b):
    shuffler = np.random.permutation(len(a))
    a = a[shuffler]
    b = b[shuffler]
    return a, b


# perceptron/delta sequential/batch learning
def update_weights(patterns, targets, weights, batch=False, delta=False):
    counter = 0
    finished = False
    delta_best = float('inf')
    delta_counter = 0
    if batch:
        batch_update = np.zeros(len(patterns[0]))
    correct = 0  # counter of correct classifications
    while not finished:

        # shuffle after each epoch
        if counter % len(patterns) == 0:
            patterns, targets = shuffle_two_arrays(patterns, targets)

        # sequential loop
        if not batch:
            for i in range(len(patterns)):

                # limit number of epochs
                if counter > N:
                    print('Number of possible epochs exceeded - training terminated')
                    finished = True
                    continue
                test_print(patterns[i])
                output = np.matmul(weights, patterns[i])
                test_print('Weighted sum =', output)
                if output > 0 and targets[i] == 1 or output <= 0 and targets[i] == -1:
                    correct += 1
                    if correct >= len(patterns):
                        finished = True
                        continue
                else:
                    correct = 0
                    if not delta:
                        if output > 0:
                            output = 1
                        else:
                            output = -1
                    delta_weight = ETA * (targets[i] - output) * patterns[i]
                    weights += delta_weight
                test_print('Weights are now:', weights)

            correct = 0

        # batch matrix method
        else:
            outputs = np.matmul(weights, patterns.T)  # W * X
            threshold_outputs = np.sign(outputs)
            delta_weight = None
            if delta:
                delta_weight = -ETA * np.matmul(outputs - targets, patterns)
            else:
                delta_weight = -ETA * np.matmul(threshold_outputs - targets, patterns)
            weights += delta_weight

            if not delta and np.equal(threshold_outputs, targets).all():
                finished = True
                continue
            else:
                if np.sum(outputs - targets) < delta_best:
                    delta_counter = 0
                    delta_best = np.sum(outputs - targets)
                else:
                    delta_counter += 1

                if delta_counter > DELTA_N:
                    finished = True
                    continue

        # count one epoch
        counter += 1

    test_print('Converged after', counter, 'epochs.')

    return weights


def get_random_weights():
    return np.random.normal(0, 0.5, 3)


def main():
    # define number of samples per class, means for each sample dimension and std
    # 3 sigma rule; probability of linear inseparability insignificant
    n = 100
    m_a = [1.5, 1.5]
    sigma_a = 0.5
    m_b = [-1.5, -1.5]
    sigma_b = 0.5

    # create class A data with bias and labels (1)
    class_a = np.array([np.random.normal(m_a[0], sigma_a, n),
                        np.random.normal(m_a[1], sigma_a, n),
                        np.ones(n)])
    test_print('Class A pattern:\n', class_a, '\n')

    # create class B data with labels (-1)
    class_b = np.array([np.random.normal(m_b[0], sigma_b, n),
                        np.random.normal(m_b[1], sigma_b, n),
                        np.ones(n)])
    test_print('Class B pattern:\n', class_b, '\n')

    # targets
    targets = np.concatenate((np.ones(n), np.ones(n) * -1), axis=0)

    # join and shuffle class A and class B together (shuffling important for sequential learning)
    patterns = np.concatenate((class_a, class_b), axis=1).T  # column-wise concatenation
    patterns, targets = shuffle_two_arrays(patterns, targets)
    test_print('All patterns (shuffled):\n', patterns, '\n')
    test_print('All targets (shuffled):\n', targets, '\n')

    plot_all(class_a, class_b, line_coefficients=None, name='samples')

    weights = get_random_weights()

    plot_all(class_a, class_b, line_coefficients=weights, name='start')

    weights = update_weights(patterns, targets, weights, batch=True, delta=True)

    plot_all(class_a, class_b, line_coefficients=weights, name='end')


if __name__ == '__main__':
    main()
