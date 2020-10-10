import matplotlib
from matplotlib import cm
from pathlib import Path

from demo.lab4.rbm import RestrictedBoltzmannMachine
from demo.lab4.util import *


class DeepBeliefNet():
    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''

    def __init__(self, sizes, image_size, n_labels, batch_size, save_path,
                 rbm_weight_decay=1e-5, rbm_momentum=0.7, rbm_learning_rate=0.01):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {

            'vis--hid': RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                   is_bottom=True, image_size=image_size, batch_size=batch_size,
                                                   visuals_save_path=os.path.join(save_path, "vis--hid"),
                                                   weight_decay=rbm_weight_decay, momentum=rbm_momentum,
                                                   learning_rate=rbm_learning_rate),

            'hid--pen': RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"],
                                                   batch_size=batch_size,
                                                   visuals_save_path=os.path.join(save_path, "hid--pen"),
                                                   weight_decay=rbm_weight_decay, momentum=rbm_momentum,
                                                   learning_rate=rbm_learning_rate),

            'pen+lbl--top': RestrictedBoltzmannMachine(ndim_visible=sizes["pen"] + sizes["lbl"],
                                                       ndim_hidden=sizes["top"],
                                                       is_top=True, n_labels=n_labels, batch_size=batch_size,
                                                       visuals_save_path=os.path.join(save_path, "pen+lbl--top"),
                                                       weight_decay=rbm_weight_decay, momentum=rbm_momentum,
                                                       learning_rate=rbm_learning_rate),
        }

        self.rbm_weight_decay = rbm_weight_decay

        self.rbm_momentum = rbm_momentum

        self.rbm_learning_rate = rbm_learning_rate

        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size

        self.n_gibbs_recog = 15

        self.n_gibbs_gener = 200

        self.n_gibbs_wakesleep = 5

        self.print_period = 2000

        self.save_path = save_path

        return

    def recognize(self, true_img, true_lbl, debug=False):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
          debug: should debugging messages be used
        """

        n_samples = true_img.shape[0]

        lbl = np.ones(true_lbl.shape) / 10.  # start the net by telling you know nothing about labels

        _, h = self.rbm_stack["vis--hid"].get_h_given_v_dir(true_img)
        _, p = self.rbm_stack["hid--pen"].get_h_given_v_dir(h)
        plp = np.concatenate((p, lbl), axis=1)
        predicted_lbls = []
        for i in range(self.n_gibbs_recog):
            _, t = self.rbm_stack["pen+lbl--top"].get_h_given_v(plp)
            plp, _ = self.rbm_stack["pen+lbl--top"].get_v_given_h(t)

            if debug or i == self.n_gibbs_recog - 1:
                predicted_lbls.append(plp[:, -true_lbl.shape[1]:])
                acc = f"accuracy={100. * np.mean(np.argmax(predicted_lbls[-1], axis=1) == np.argmax(true_lbl, axis=1)):.3f}%"
                print(acc)
                if debug:
                    with open(os.path.join(self.save_path, f"recognize_len={n_samples}.txt"), "a") as f:
                        f.write(f"i={i} {acc}\n{predicted_lbls[-1][:100].tolist()}\n")

        return predicted_lbls

    def generate(self, true_lbl, name):

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        n_sample = true_lbl.shape[0]

        records = []
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]);
        ax.set_yticks([])

        # pen_sample = np.random.rand(n_sample, self.sizes["pen"])  # 1
        # plp_from_bias, _ = self.rbm_stack["pen+lbl--top"].get_v_given_h(np.zeros((n_sample, self.sizes["top"])))  # 1'
        # pen_sample = plp_from_bias[:, :-true_lbl.shape[1]] # 1'
        pen_sample_probs = sigmoid(
            np.zeros((n_sample, self.sizes["pen"])) + self.rbm_stack["pen+lbl--top"].bias_v[:-true_lbl.shape[1]])  # 2
        pen_sample = sample_binary(pen_sample_probs)  # 2
        # pen_sample = pen_sample_probs # 2'
        # TODO pen_sample can be:
        #       1. random
        #       2. sample from biases
        #       3. sample drawn from distro obtained by propagating random image all the way from the input
        plp = np.concatenate((pen_sample, true_lbl), axis=1)
        for _ in range(self.n_gibbs_gener):
            plp[:, -true_lbl.shape[1]:] = true_lbl  # clamping of lables
            _, t = self.rbm_stack["pen+lbl--top"].get_h_given_v(plp)
            plp, lp = self.rbm_stack["pen+lbl--top"].get_v_given_h(t)

            p = lp[:, :-true_lbl.shape[1]]
            _, h = self.rbm_stack["hid--pen"].get_v_given_h_dir(p)
            pv, _ = self.rbm_stack["vis--hid"].get_v_given_h_dir(h)
            records.append([ax.imshow(pv.reshape(self.image_size), cmap=cm.viridis, vmin=0, vmax=1, animated=True,
                                      interpolation=None)])

        file_prefix = os.path.join(self.save_path, "%s.generate%d." % (name, np.argmax(true_lbl)))
        writer, ext = matplotlib.animation.PillowWriter(fps=30), "gif"
        # writer, ext = matplotlib.animation.FFMpegWriter(fps=30), "mp4"
        stitch_video(fig, records).save(file_prefix + ext, writer=writer)
        ax.imshow(pv.reshape(self.image_size), cmap=cm.viridis, vmin=0, vmax=1, animated=True, interpolation=None)
        plt.savefig(file_prefix + "png", dpi=300)
        plt.close('all')
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try:
            self.loadfromfile_rbm(loc=os.path.join(self.save_path, "trained_rbm"), name="vis--hid")
        except IOError:
            print("training vis--hid")
            self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)
            self.savetofile_rbm(loc=os.path.join(self.save_path, "trained_rbm"), name="vis--hid")

        self.rbm_stack["vis--hid"].untwine_weights()
        _, h = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)
        # TODO or should I take probs

        try:
            self.loadfromfile_rbm(loc=os.path.join(self.save_path, "trained_rbm"), name="hid--pen")
        except IOError:
            print("training hid--pen")
            self.rbm_stack["hid--pen"].cd1(h, n_iterations)
            self.savetofile_rbm(loc=os.path.join(self.save_path, "trained_rbm"), name="hid--pen")

        self.rbm_stack["hid--pen"].untwine_weights()
        _, p = self.rbm_stack["hid--pen"].get_h_given_v_dir(h)
        lp = np.concatenate((p, lbl_trainset), axis=1)
        # TODO or should I take probs

        try:
            self.loadfromfile_rbm(loc=os.path.join(self.save_path, "trained_rbm"), name="pen+lbl--top")
        except IOError:
            print("training pen+lbl--top")
            self.rbm_stack["pen+lbl--top"].cd1(lp, n_iterations)
            self.savetofile_rbm(loc=os.path.join(self.save_path, "trained_rbm"), name="pen+lbl--top")

        return

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("\ntraining wake-sleep..")

        try:

            self.loadfromfile_dbn(loc=os.path.join(self.save_path, "trained_dbn"), name="vis--hid")
            self.loadfromfile_dbn(loc=os.path.join(self.save_path, "trained_dbn"), name="hid--pen")
            self.loadfromfile_rbm(loc=os.path.join(self.save_path, "trained_dbn"), name="pen+lbl--top")

        except IOError:

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):

                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.

                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.

                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

                if it % self.print_period == 0: print("iteration=%7d" % it)

            self.savetofile_dbn(loc=os.path.join(self.save_path, "trained_dbn"), name="vis--hid")
            self.savetofile_dbn(loc=os.path.join(self.save_path, "trained_dbn"), name="hid--pen")
            self.savetofile_rbm(loc=os.path.join(self.save_path, "trained_dbn"), name="pen+lbl--top")

        return

    def loadfromfile_rbm(self, loc, name):
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_rbm(self, loc, name):
        if not Path(loc).is_dir():
            Path(loc).mkdir(parents=True, exist_ok=False)
        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self, loc, name):

        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy" % (loc, name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/dbn.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/dbn.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_dbn(self, loc, name):

        if not Path(loc).is_dir():
            Path(loc).mkdir(parents=True, exist_ok=False)
        np.save("%s/dbn.%s.weight_v_to_h" % (loc, name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v" % (loc, name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return
