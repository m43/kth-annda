from tqdm import tqdm

from demo.lab4.util import *
from utils.util import ensure_dir, plot_metric

plt.style.use('seaborn-white')


class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''

    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=(28, 28), is_top=False, n_labels=10,
                 batch_size=10, visuals_save_path="", weight_decay=0.0001, learning_rate=0.01, momentum=0.7):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
          visuals_save_path: the folder where the debug images should be saved
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom: self.image_size = image_size

        self.is_top = is_top

        if is_top: self.n_labels = n_labels

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = learning_rate

        self.momentum = momentum

        self.weight_decay = weight_decay

        self.print_period = 5000

        ensure_dir(visuals_save_path)
        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            "period": 5000,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            "ids": np.random.randint(0, self.ndim_hidden, 25),  # pick some random hidden units
            "path": visuals_save_path
        }

        return

    def viz_all_rf(self, name="rf"):
        """
        Visualize all receptive fields and save
        """
        weights = self.weight_vh.reshape((self.image_size[0], self.image_size[1], -1))
        imax = abs(weights).max()
        for hw in range(weights.shape[2]):
            fig, axs = plt.subplots()
            plt.title(f"RF {hw}")
            axs.set_xticks([])
            axs.set_yticks([])
            axs.imshow(weights[:, :, hw], cmap="bwr", vmin=-imax, vmax=imax, interpolation=None)
            plt.savefig(os.path.join(self.rf["path"], f"{name}_{hw:04d}.png"), dpi=200)
            plt.close('all')

    def viz_weights_histogram(self, name):
        fig, axs = plt.subplots(1, 3, figsize=(6.4 * 3, 4.8))
        fig.suptitle(f'Histogram plot of weights and biases, {name}')

        for ax in axs.flat:
            ax.set(xlabel='weight value', ylabel='count')

        kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=100)

        axs[0].set_title('weights vh')
        axs[0].hist(self.weight_vh.flatten(), label='weights', **kwargs)

        axs[1].set_title('bias v')
        axs[1].hist(self.bias_v.flatten(), **kwargs)

        axs[2].set_title('bias h')
        axs[2].hist(self.bias_h.flatten(), **kwargs)

        plt.savefig(os.path.join(self.rf["path"], f"weights_histogram_{name}.png"), dpi=200)
        plt.close()

    def cd1(self, visible_trainset, n_iterations=10000):

        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print(f"learning CD1. visuals_save_path={self.rf['path']}")

        logs = {}
        logs["recon_losses"] = []
        logs["sum_of_weights_changes"] = [[], [], [], "w_vh, b_v, b_h"]
        for it in tqdm(range(n_iterations)):
            mini_idx = it % (visible_trainset.shape[0] // self.batch_size)
            if mini_idx == 0:
                shuffle_indices = np.arange(visible_trainset.shape[0])
                np.random.shuffle(shuffle_indices)
                visible_trainset = visible_trainset[shuffle_indices]

            # mini_indices = np.random.choice(visible_trainset.shape[0], self.batch_size, replace=False)
            # pv_0 = v0 = visible_trainset[mini_indices]
            pv_0 = v0 = visible_trainset[self.batch_size * mini_idx:self.batch_size * (mini_idx + 1)]
            ph_0, h0 = self.get_h_given_v(v0)
            p_v1, _ = self.get_v_given_h(h0)
            p_h1, _ = self.get_h_given_v(p_v1)

            self.update_params(pv_0, ph_0, p_v1, p_h1)

            # visualize once in a while when visible layer is input images

            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape((self.image_size[0], self.image_size[1], -1)),
                       it=it, grid=self.rf["grid"], save_path=self.rf["path"])

            # print progress

            if it % self.print_period == 0:
                _, hr = self.get_h_given_v(visible_trainset)  # reconstruction loss
                _, vr = self.get_v_given_h(hr)
                recon_loss = np.linalg.norm(visible_trainset - vr) / visible_trainset.shape[0]

                logs["recon_losses"].append(recon_loss)
                logs["sum_of_weights_changes"][0].append(np.abs(self.delta_weight_vh).sum())
                logs["sum_of_weights_changes"][1].append(np.abs(self.delta_bias_v).sum())
                logs["sum_of_weights_changes"][2].append(np.abs(self.delta_bias_h).sum())
                print(logs)
                print("iteration=%7d recon_loss=%4.10f" % (it, recon_loss))

                self.viz_weights_histogram(f"{it:06d}")

        with open(os.path.join(self.rf["path"], f"logs.txt"), "w") as f:
            f.write(f"{logs}")
        plot_metric(logs["recon_losses"], os.path.join(self.rf["path"], f"recon_loss"), True)

        return logs

    def update_params(self, v_0, h_0, v_k, h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        self.delta_bias_v = self.momentum * self.delta_bias_v + self.learning_rate * np.average(v_0 - v_k, axis=0)
        self.delta_weight_vh = self.learning_rate * ((v_0.T @ h_0 - v_k.T @ h_k) / v_0.shape[
            0] - self.weight_decay * self.weight_vh) + self.momentum * self.delta_weight_vh
        # self.learning_rate * (np.average(v_0[:, :, np.newaxis] * h_0[:, np.newaxis, :], axis=0)
        # - np.average(v_k[:, :, np.newaxis] * h_k[:, np.newaxis, :], axis=0))
        self.delta_bias_h = self.momentum * self.delta_bias_h + self.learning_rate * np.average(h_0 - h_k, axis=0)

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

        return

    def get_h_given_v(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        return self._get_h_given_v(visible_minibatch, self.weight_vh)

    def get_v_given_h(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """
        return self._get_v_given_h(hidden_minibatch, self.weight_vh.T)

    def _get_h_given_v(self, visible_minibatch, weights):
        assert weights is not None

        total_input_for_each_unit = (visible_minibatch @ weights) + self.bias_h
        p_h_given_v = sigmoid(total_input_for_each_unit)
        h_sample = sample_binary(p_h_given_v)

        return p_h_given_v, h_sample

    def _get_v_given_h(self, hidden_minibatch, weights):
        assert weights is not None

        total_input_for_each_unit = (hidden_minibatch @ weights) + self.bias_v
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            left_in, labels_in = total_input_for_each_unit[:, :-self.n_labels], total_input_for_each_unit[:,
                                                                                -self.n_labels:]
            left_p_v_given_h, labels_p_v_given_h = sigmoid(left_in), softmax(labels_in)
            left_v_sample, labels_v_sample = sample_binary(left_p_v_given_h), sample_categorical(labels_p_v_given_h)

            p_v_given_h = np.concatenate((left_p_v_given_h, labels_p_v_given_h), axis=1)
            v_sample = np.concatenate((left_v_sample, labels_v_sample), axis=1)

        else:

            p_v_given_h = sigmoid(total_input_for_each_unit)
            v_sample = sample_binary(p_v_given_h)

        return p_v_given_h, v_sample

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):

        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))
        self.weight_vh = None

    def get_h_given_v_dir(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        return self._get_h_given_v(visible_minibatch, self.weight_v_to_h)

    def get_v_given_h_dir(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        return self._get_v_given_h(hidden_minibatch, self.weight_h_to_v)

    def update_generate_params(self, inps, trgs, preds):

        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v

        return

    def update_recognize_params(self, inps, trgs, preds):

        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return
