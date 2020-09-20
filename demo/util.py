import math

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d as art3d

from utils.util import TwoClassDatasetGenerator


def perpare_reproducable_separable_dataset(seed=72):
    return TwoClassDatasetGenerator(
        m_a=(1.5, 1.5), n_a=100, sigma_a=(0.5, 0.5),
        m_b=(-1.5, -1.5), n_b=100, sigma_b=(0.5, 0.5)
    ).random_1(seed=72)


def perpare_reproducable_separable_dataset_impossible_with_no_bias(seed=72):
    TwoClassDatasetGenerator(
        m_a=(1, 1), n_a=100, sigma_a=(0.5, 1),
        m_b=(4, 4), n_b=100, sigma_b=(0.6, 0.3)
    ).random_1(seed=seed)


def perpare_reproducable_inseparable_dataset_1(seed=72):
    return TwoClassDatasetGenerator(
        m_a=(1.8, 1.8), n_a=100, sigma_a=(1, 1),
        m_b=(0, 0), n_b=100, sigma_b=(0.5, 0.7)
    ).random_1(seed=seed)


def perpare_reproducable_inseparable_dataset_2_with_subsets(seed=72):
    # Get the original dataset from which will subsets be taken
    inputs, targets = TwoClassDatasetGenerator(
        m_a=(1.0, 0.3), n_a=100, sigma_a=(0.2, 0.2),
        m_b=(0.0, -0.1), n_b=100, sigma_b=(0.3, 0.3)
    ).random_2(seed=seed)

    # Prepare 4th subsample
    idx_4 = []
    p_a_below = 0.2
    p_a_above = 0.8
    for i, t in enumerate(targets[0]):
        if t == 1:
            if inputs[0, i] < 0 and np.random.random() < p_a_below:
                idx_4.append(i)
            if inputs[0, i] > 0 and np.random.random() < p_a_above:
                idx_4.append(i)

    # Create subsets list and a list of their negations
    subsets, negated_subsets = [], []
    for subset, subset_negation in [
        sample_two_class_dataset(inputs, targets, 25, 25),
        sample_two_class_dataset(inputs, targets, 50, 0),
        sample_two_class_dataset(inputs, targets, 0, 50),
        ((np.delete(inputs, idx_4, axis=1), np.delete(targets, idx_4, axis=1)), (inputs[:, idx_4], targets[:, idx_4]))
    ]:
        subsets.append(subset)
        negated_subsets.append(subset_negation)

    return (inputs, targets), subsets, negated_subsets


def print_results_as_table(results, keys):
    print("", end="\t")
    for k, v in results.items():
        print(k, end="\t")
    print()
    for key in keys:
        print(f"{key}_mean", end="\t")
        for k, v in results.items():
            print(v[key][0], end="\t")
        print()
        print(f"{key}_std", end="\t")
        for k, v in results.items():
            print(v[key][1], end="\t")
        print()


def two_class_conf_mat_metrics(cm):
    tp, tn, fp, fn = cm[0, 0], cm[1, 1], cm[0, 1], cm[1, 0]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return accuracy, sensitivity, specificity, precision, recall


def sample_two_class_dataset(inputs, targets, n_a, n_b):
    idx_1 = []
    for i, t in enumerate(targets[0]):
        if t == 1 and n_a > 0:
            idx_1.append(i)
            n_a -= 1
        if t != 1 and n_b > 0:
            idx_1.append(i)
            n_b -= 1
    return (np.delete(inputs, idx_1, axis=1), np.delete(targets, idx_1, axis=1)), (inputs[:, idx_1], targets[:, idx_1])


def prepare_dataset(function):
    inputs = [[i / 100.0] for i in range(0, int(2 * math.pi * 100), 10)]
    train_inputs, valid_inputs = np.array(inputs[::2] + inputs[1::4]), np.array(inputs[3::4])
    train_targets = np.array([[function(x)] for x in train_inputs])
    valid_targets = np.array([[function(x)] for x in valid_inputs])

    test_inputs = np.array([[i / 100.0] for i in range(5, int(2 * math.pi * 100), 10)])
    test_targets = np.array([[function(x)] for x in test_inputs])

    return (train_inputs, train_targets), (valid_inputs, valid_targets), (test_inputs, test_targets)


def plot_ballist(x, y, z1, z2, clusters, centres_x, centres_y, sigmas, save_filename="", show_plot=True,
                 title="Ballist dataset: <angle, velocity> --> <distance, height>"):
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)
    ax1 = fig.add_subplot(221, projection="3d")
    for cx, cy, cs, cluster in zip(centres_x, centres_y, sigmas, range(len(centres_x))):
        c = Circle((cx, cy), cs, color=cm.rainbow(cluster / (len(centres_x) - 1)), alpha=0.3)
        ax1.add_patch(c)
        art3d.pathpatch_2d_to_3d(c, z=0, zdir="z")
    ax1.scatter(centres_x, centres_y, c="black", linewidth=1, marker="o", alpha=1)
    ax1.scatter(x, y, z1, c=clusters, cmap=cm.rainbow, linewidth=3, alpha=1)
    ax1.set_xlabel('angle')
    ax1.set_ylabel('velocity')
    ax1.set_zlabel('distance')
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.plot_trisurf(x, y, z1, cmap=cm.rainbow, edgecolor='none')
    ax2.set_xlabel('angle')
    ax2.set_ylabel('velocity')
    ax2.set_zlabel('distance')
    ax3 = fig.add_subplot(223, projection="3d")
    for cx, cy, cs, cluster in zip(centres_x, centres_y, sigmas, range(len(centres_x))):
        c = Circle((cx, cy), cs, color=cm.viridis(cluster / (len(centres_x) - 1)), alpha=0.3)
        ax3.add_patch(c)
        art3d.pathpatch_2d_to_3d(c, z=0, zdir="z")
    ax3.scatter(centres_x, centres_y, c="black", linewidth=1, marker="o", alpha=1)
    ax3.scatter(x, y, z2, c=clusters, cmap=cm.viridis, linewidth=3, alpha=1)
    ax3.set_xlabel('angle')
    ax3.set_ylabel('velocity')
    ax3.set_zlabel('height')
    ax4 = fig.add_subplot(224, projection="3d")
    ax4.plot_trisurf(x, y, z2, cmap=cm.viridis, edgecolor='none')
    # ax4.scatter(x, y, z2, c=clusters, cmap=cm.viridis, linewidth=3)
    ax4.set_xlabel('angle')
    ax4.set_ylabel('velocity')
    ax4.set_zlabel('height')
    if save_filename:
        plt.savefig(save_filename, dpi=300)
    if show_plot:
        plt.show()
    plt.close()
