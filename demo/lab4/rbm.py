# from demo.lab4.util import *
from util import *



class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''

    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28, 28], is_top=False, n_labels=10,
                 batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
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

        self.learning_rate = 0.01

        self.momentum = 0.7

        self.print_period = 2

        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            "period": 2,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            "ids": np.random.randint(0, self.ndim_hidden, 25)  # pick some random hidden units
        }

        return



    def gibbs(self, v_0, k):
        _, h_0 = self.get_h_given_v(v_0)
        v_k = v_0.copy()
        h_k = h_0.copy()
        for i in range(k):
            _, h_k = self.get_h_given_v(v_k)
            _, v_k = self.get_v_given_h(h_k)
        return h_0, h_k, v_k 

    def gibbs_p(self, v_0, k):
        ph_0, h_k = self.get_h_given_v(v_0)
        for i in range(k):
            pv_k, _ = self.get_v_given_h(h_k)
            ph_k, h_k = self.get_h_given_v(pv_k)
        return ph_0, pv_k, ph_k

    def cd1(self, visible_trainset, n_iterations=10):
        print("learning CD1")

        recon_losses = []    
        self.nprint = 0
        n_samples = visible_trainset.shape[0]
        trainset = visible_trainset.copy()

        for it in range(n_iterations):
            np.random.shuffle(trainset)
            print("epoch ", it, " ---")
            for i in range(int(n_samples / self.batch_size)):
                v_0 = trainset[int(i * self.batch_size):int((i+1) * self.batch_size)]
                ph_0, pv_k, ph_k = self.gibbs_p(v_0, 1)
                self.update_params(v_0, ph_0, pv_k, ph_k)

            # visualize once in a while when visible layer is input images
            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape((self.image_size[0], self.image_size[1], -1)),
                       it=it, grid=self.rf["grid"])

            # if it % self.print_period == 0:
            _, hr  = self.get_h_given_v(trainset) # reconstruction loss
            _, vr  = self.get_v_given_h(hr)
            recon_losses.append((1/n_samples) * np.linalg.norm(trainset - vr))

            # print("iteration=%7d recon_loss=%4.4f" % (it, (1/n_samples) * np.linalg.norm(visible_trainset - vr)))
        return recon_losses

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

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters
        

        n_samples = v_0.shape[0]
        # print(v_0.shape) # 10x784
        # print(h_0.shape) # 10x200
        # print(v_k.shape) # 10x784
        # print(h_k.shape) # 10x200

        self.delta_weight_vh = (v_0.T @ h_0 - v_k.T @ h_k) / n_samples
        self.delta_bias_h = np.average((h_0 - h_k), axis=0)
        self.delta_bias_v = np.average((v_0 - v_k), axis=0)

        self.bias_v += self.learning_rate * self.delta_bias_v
        self.weight_vh += self.learning_rate * self.delta_weight_vh
        self.bias_h += self.learning_rate * self.delta_bias_h

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

        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        p_h = sigmoid(self.bias_h + (visible_minibatch @ self.weight_vh)) # 10 x 200        
        h = sample_binary(p_h)

        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below) 

        return p_h, h
        # return np.zeros((n_samples, self.ndim_hidden)), np.zeros((n_samples, self.ndim_hidden))

    def get_v_given_h(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]


        support = self.bias_v + (hidden_minibatch @ self.weight_vh.T)

 

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            

                                    # 500                   # 10
            sup1, sup2 = support[:, :-self.n_labels], support[:, -self.n_labels:]
            p_s2 = softmax(sup2)            
            v_s2 = sample_categorical(p_s2)

            p_s1 = sigmoid(sup1)
            v_s1 = sample_binary(p_s1)

            # p_v, v = np.zeros((n_samples, self.ndim_visible)), np.zeros((n_samples, self.ndim_visible))

            p_v = np.concatenate((p_s1, p_s2), axis=1)
            v = np.concatenate((v_s1, v_s2), axis=1)


        else:

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)             
            
            # print(hidden_minibatch.shape) # (10, 200)
            # print(self.weight_vh.shape) # (784, 200)
            # print(self.bias_v.shape) # (784,)
            p_v = sigmoid(support)
            v = sample_binary(p_v)

        return p_v, v        
        # return np.zeros((n_samples, self.ndim_visible)), np.zeros((n_samples, self.ndim_visible))

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

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 

        p_h = sigmoid(self.bias_h + (visible_minibatch @ self.weight_v_to_h))        
        h = sample_binary(p_h)


        return p_h, h
        # return np.zeros((n_samples, self.ndim_hidden)), np.zeros((n_samples, self.ndim_hidden))

    def get_v_given_h_dir(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        support = self.bias_v + (hidden_minibatch @ self.weight_h_to_v)


        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)

            sup1, sup2 = support[:, :-self.n_labels], support[:, -self.n_labels:]
            p_s2 = softmax(sup2)            
            v_s2 = sample_categorical(p_s2)

            p_s1 = sigmoid(sup1)
            v_s1 = sample_binary(p_s1)

            p_v = np.concatenate((p_s1, p_s2), axis=1)
            v = np.concatenate((v_s1, v_s2), axis=1)

        else:

            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             
            p_v = sigmoid(support)        
            v = sample_binary(p_v)

        return p_v, v
        # return np.zeros((n_samples, self.ndim_visible)), np.zeros((n_samples, self.ndim_visible))

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
