import math
import os

from model.delta_rule_perceptron import DeltaRulePerceptron
from model.perceptron_rule_perceptron import PerceptronRulePerceptron
from utils.util import scatter_plot_2d_features, TwoClassDatasetGenerator


def perpare_reproducable_separable_dataset():
    return TwoClassDatasetGenerator(
        m_a=(1.5, 1.5), n_a=100, sigma_a=(0.5, 0.5),
        m_b=(-1.5, -1.5), n_b=100, sigma_b=(0.5, 0.5)
    ).random(seed=72)


def perceptron_learning_demo(inputs, targets, name, debug, save_folder, max_iter, eta):
    if debug:
        print(f"{name} started")

    perceptron = PerceptronRulePerceptron(inputs, targets, debug, os.path.join(save_folder, name))
    if debug:
        print(f"Before training - Acc: {perceptron.eval(inputs, targets)}")
        scatter_plot_2d_features(inputs, targets, name + "_START", save_folder, perceptron.W)

    try:
        weights_per_epoch, acc, convergence_epoch = perceptron.train(inputs, targets, eta, max_iter=max_iter,
                                                                     shuffle=True)
    except RuntimeError:
        print("RUNTIME ERROR in train...")
        return math.nan, math.nan

    if debug:
        print(f"After training - Acc: {acc} CEpoch: {convergence_epoch}")
        scatter_plot_2d_features(inputs, targets, name + "_END", save_folder, perceptron.W)
        # TODO animate weights_per_epoch...
        print()

    return acc, convergence_epoch


def delta_rule_learning_demo(inputs, targets, name, debug, save_folder, max_iter, eta, delta_n, batch_size, bias):
    if debug:
        print(f"{name} started")

    perceptron = DeltaRulePerceptron(inputs, targets, debug, os.path.join(save_folder, name), bias=bias)
    if debug:
        acc, loss = perceptron.eval(inputs, targets)
        print(f"Before training - Acc: {acc} Loss: {loss}")
        scatter_plot_2d_features(inputs, targets, name + "_START", save_folder, perceptron.W)

    try:
        weights_per_epoch, acc, loss, convergence_epoch = perceptron.train(inputs, targets, eta, max_iter, batch_size,
                                                                           shuffle=True, stop_after=delta_n)
    except RuntimeError:
        print("RUNTIME ERROR in train...")
        return math.nan, math.nan, math.nan

    if debug:
        print(f"After training - Acc: {acc} Loss: {loss} CEpoch: {convergence_epoch}")
        scatter_plot_2d_features(inputs, targets, name + "_END", save_folder, perceptron.W)
        # TODO animate weights_per_epoch...
        print()

    return acc, loss, convergence_epoch


def print_results_as_table(results, keys):
    for k, v in results.items():
        print(k, end="\t")
    print()
    for key in keys:
        for k, v in results.items():
            print(v[key][0], end="\t")
        print()
        for k, v in results.items():
            print(v[key][1], end="\t")
        print()


if __name__ == '__main__':
    # import numpy as np
    # import matplotlib
    # import matplotlib.pyplot as plt
    # from matplotlib.animation import FuncAnimation
    # matplotlib.use("TkAgg")
    #
    # x_data = []
    # y_data = []
    #
    # fig, ax = plt.subplots()
    # ax.set_xlim(0, 105)
    # ax.set_ylim(0, 12)
    # line, = ax.plot(0, 0)
    #
    #
    # def animation_frame(i):
    #     x_data.append(i * 10)
    #     y_data.append(i)
    #
    #     line.set_xdata(x_data)
    #     line.set_ydata(y_data)
    #     return line,
    #
    #
    # animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 10, 0.1), interval=10)
    # plt.show()

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation


    def update_line(num, data, line):
        line.set_data(data[..., :num])
        return line,


    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig1 = plt.figure()

    data = np.random.rand(2, 25)
    l, = plt.plot([], [], 'r-')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.title('test')
    line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                       interval=50, blit=True)
    line_ani.save('lines.mp4', writer=writer)

    fig2 = plt.figure()

    x = np.arange(-9, 10)
    y = np.arange(-9, 10).reshape(-1, 1)
    base = np.hypot(x, y)
    ims = []
    for add in np.arange(15):
        ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                       blit=True)
    im_ani.save('im.mp4', writer=writer)
