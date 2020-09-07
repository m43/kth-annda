import numpy as np

from utils.util import TwoClassDatasetGenerator


def perpare_reproducable_separable_dataset():
    return TwoClassDatasetGenerator(
        m_a=(1.5, 1.5), n_a=100, sigma_a=(0.5, 0.5),
        m_b=(-1.5, -1.5), n_b=100, sigma_b=(0.5, 0.5)
    ).random_1(seed=72)


def perpare_reproducable_separable_dataset_impossible_with_no_bias():
    TwoClassDatasetGenerator(
        m_a=(1, 1), n_a=100, sigma_a=(0.5, 1),
        m_b=(4, 4), n_b=100, sigma_b=(0.6, 0.3)
    ).random_1(seed=72)


def perpare_reproducable_inseparable_dataset_1():
    return TwoClassDatasetGenerator(
        m_a=(1.8, 1.8), n_a=100, sigma_a=(1, 1),
        m_b=(0, 0), n_b=100, sigma_b=(0.5, 0.7)
    ).random_1(seed=72)


def perpare_reproducable_inseparable_dataset_2_with_subsets():
    # Get the original dataset from which will subsets be taken
    inputs, targets = TwoClassDatasetGenerator(
        m_a=(1.0, 0.3), n_a=100, sigma_a=(0.2, 0.2),
        m_b=(0.0, -0.1), n_b=100, sigma_b=(0.3, 0.3)
    ).random_2(seed=72)

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
