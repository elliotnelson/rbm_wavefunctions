import tensorflow as tf
import itertools as it
import numpy as np

class RBM(object):
    
    ''' Restricted Boltzmann Machine '''
    
    def __init__(self, num_hidden, num_visible, num_samples=128, weights=None, visible_bias=None, hidden_bias=None):
        ''' Constructor '''
        # number of hidden units
        self.num_hidden = num_hidden
        # number of visible units
        self.num_visible = num_visible

        # visible bias
        default = tf.random_normal(shape=(self.num_visible, 1), mean=-7.4947795405533468901, stddev=0.00000001)  # ELN: (mean,stddev) default = (-0.5,0.05)
        # default = tf.random_normal(shape=(self.num_visible, 1), mean=-0.0, stddev=0.05)
        # default = tf.zeros(shape=(self.num_visible, 1))
        self.visible_bias = self._create_parameter_variable(visible_bias, default)
        self.visible_bias_frozen = self.visible_bias
        # hidden bias
        default = tf.random_normal(shape=(self.num_hidden, 1), mean=-24.179888055607690930, stddev=0.00000001)  # ELN: (mean,stddev) default = (-0.2,0.05)
        # default = tf.random_normal(shape=(self.num_hidden, 1), mean=-0.0, stddev=0.05)
        # default = tf.zeros(shape=(self.num_hidden, 1))
        self.hidden_bias = self._create_parameter_variable(hidden_bias, default)
        self.hidden_bias_frozen = self.hidden_bias
        # pairwise weights
        default = tf.random_normal(shape=(self.num_visible, self.num_hidden), mean=15.837174845870938954, stddev=0.00000001)  # ELN: somehow cost_kl is most negative near w=13
        # default = tf.random_normal(shape=(self.num_visible, self.num_hidden), mean=0.0, stddev=0.05)
        # default = tf.zeros(shape=(self.num_visible, self.num_hidden))
        self.weights = self._create_parameter_variable(weights, default)
        self.weights_frozen = self.weights

        # ELN: adding "frozen" variables above, which are currently trainable, but explicitly not included in optimizer.minimize() in main.py

        # THIS DIDN'T WORK:
        # self.weights_frozen = tf.get_variable('weights_frozen', [num_visible,num_hidden])
        # tf.assign(self.weights_frozen, self.weights)
        # self.visible_bias_frozen = tf.get_variable('visible_bias_frozen', [num_visible,1])
        # tf.assign(self.visible_bias_frozen, self.visible_bias)

        # Variables for sampling.
        # number of samples to return when sample_p_of_v is called.
        self.num_samples = num_samples

        self.hidden_samples = tf.Variable(
            self.sample_binary_tensor(tf.constant(0.5), self.num_samples, self.num_hidden),
            trainable=False, name='hidden_samples'
        )

        # ELN: adding
        self.visible_samples_model = tf.Variable(
            self.sample_binary_tensor(tf.constant(0.5), self.num_samples, self.num_visible),
            trainable=False, name='visible_samples_model'
        )

        self.p_of_v = None
        self._all_hidden_states = None
        self._all_visible_states = None   # ELN: adding
        self.max_feasible_for_log_pf = 24

    @property
    def all_hidden_states(self):
        ''' Build array with all possible configuration of the hidden layer '''
        if self._all_hidden_states is None:
            assert self.num_hidden <= self.max_feasible_for_log_pf, \
                'cannot generate all hidden states for num_hidden > {}'.format(self.max_feasible_for_log_pf)
            self._all_hidden_states = np.array(list(it.product([0, 1], repeat=self.num_hidden)), dtype=np.float32)
        return self._all_hidden_states

    # ELN: adding, modified from all_hidden_states() above
    @property
    def all_visible_states(self):
        ''' Build array with all possible configuration of the visible layer '''
        if self._all_visible_states is None:
            assert self.num_visible <= self.max_feasible_for_log_pf, \
                'cannot generate all visible states for num_visible > {}'.format(self.max_feasible_for_log_pf)
            self._all_visible_states = np.array(list(it.product([0, 1], repeat=self.num_visible)), dtype=np.float32)
        return self._all_visible_states

    @staticmethod
    def _create_parameter_variable(initial_value=None, default=None):
        ''' Initialize variables '''
        if initial_value is None:
            initial_value = default
        return tf.Variable(initial_value)

     
    def p_of_h_given(self, v, frozen_params=False):
        ''' Conditional probability of hidden layer given visible state '''
        # type: (tf.Tensor) -> tf.Tensor

        if frozen_params==True:
            weights = self.weights_frozen
            hidden_bias = self.hidden_bias_frozen
        else:
            weights = self.weights
            hidden_bias = self.hidden_bias

        return tf.nn.sigmoid(tf.matmul(v, weights) + tf.transpose(hidden_bias))

    def p_of_v_given(self, h):
        ''' Conditional probability of visible layer given hidden state '''
        # type: (tf.Tensor) -> tf.Tensor
        return tf.nn.sigmoid(tf.matmul(h, self.weights, transpose_b=True) + tf.transpose(self.visible_bias))

    def sample_h_given(self, v):
        ''' Sample the hidden nodes given a visible state '''
        # type: (tf.Tensor) -> (tf.Tensor, tf.Tensor)
        b = tf.shape(v)[0]  # number of samples
        m = self.num_hidden
        prob_h = self.p_of_h_given(v)
        samples = self.sample_binary_tensor(prob_h, b, m)
        return samples, prob_h

    def sample_v_given(self, h):
        ''' Samples the visible nodes given a hidden state '''
        # type: (tf.Tensor) -> (tf.Tensor, tf.Tensor)
        b = tf.shape(h)[0]  # number rof samples
        n = self.num_visible
        prob_v = self.p_of_v_given(h)
        samples = self.sample_binary_tensor(prob_v, b, n)
        return samples, prob_v

    # ELN: adding visible_samples argument in case of initializing chains with visible samples
    def stochastic_maximum_likelihood(self, num_iterations, visible_samples=None):
        # type: (int) -> (tf.Tensor, tf.Tensor, tf.Tensor)
        """
        Define persistent CD_k. Stores the results of `num_iterations` of contrastive divergence in
        class variables.
        :param int num_iterations: The 'k' in CD_k.
        """
        h_samples = self.hidden_samples
        v_samples = None
        p_of_v = 0

        # ELN: option for (non-persistent) CD_k
        if self.not_pcd==1:
            v_samples = visible_samples
            h_samples, _ = self.sample_h_given(v_samples)
            num_iterations -= 1  # ELN: adjusts num_iterations since we initialize h_samples with 1 iteration here

        for i in range(num_iterations):
            v_samples, p_of_v = self.sample_v_given(h_samples)
            h_samples, _ = self.sample_h_given(v_samples)

        self.hidden_samples = self.hidden_samples.assign(h_samples)
        # ELN: replacing the RHS above with h_samples seemed to screw up learning and prevent RBM from finding approximate GHZ solution
        self.p_of_v = p_of_v

        # ELN: adding modified version using conditional probabilities for negative phase "fantasy particles" -> sigmoid function
        if self.gibbs_chains_end_prob==1:
            # v_samples = tf.sigmoid(self.visible_bias + tf.matmul(self.weights, self.hidden_samples, transpose_b=True))
            v_samples = p_of_v

        # ELN: adding this to store visible samples for negative phase, IF we calculate negative phase that way
        # self.visible_samples_model = self.visible_samples_model.assign(v_samples)  # v_samples

        return self.hidden_samples, v_samples
    
    def energy(self, hidden_samples, visible_samples, frozen_params=False):
        # type: (tf.Tensor, tf.Tensor) -> tf.Tensor
        """Compute the energy:
            E = - aT*v - bT*h - vT*W*h.

        Note that since we want to support larger batch sizes, we do element-wise multiplication between
        vT*W and h, and sum along the columns to get a Tensor of shape batch_size by 1

        :param hidden_samples: Tensor of shape batch_size by num_hidden
        :param visible_samples:  Tensor of shae batch_size by num_visible
        """

        if frozen_params==True:
            weights = self.weights_frozen
            visible_bias = self.visible_bias_frozen
            hidden_bias = self.hidden_bias_frozen
        else:
            weights = self.weights
            visible_bias = self.visible_bias
            hidden_bias = self.hidden_bias

        return (-tf.matmul(hidden_samples, hidden_bias)  # b x m * m x 1
                - tf.matmul(visible_samples, visible_bias)  # b x n * n x 1
                    - tf.reduce_sum(tf.matmul(visible_samples, weights) * hidden_samples, 1, keep_dims=True))
    # ELN: I added keep_dims argument; note that the * does the same thing as tf.multiply(), so alternate line =
    #             - tf.reduce_sum(tf.multiply(tf.matmul(visible_samples, weights), hidden_samples), 1))
 
    def free_energy(self, visible_samples, frozen_params=False):
        ''' Compute the free energy:
            F = - aT*v - sum(softplus(a + vT*W)) '''  # ELN: adding a - sign here

        if frozen_params==True:
            weights = self.weights_frozen
            visible_bias = self.visible_bias_frozen
            hidden_bias = self.hidden_bias_frozen
        else:
            weights = self.weights
            visible_bias = self.visible_bias
            hidden_bias = self.hidden_bias

        # ELN: adding - sign
        free_energy = - (tf.matmul(visible_samples, visible_bias)
                         + tf.reduce_sum(tf.nn.softplus(tf.matmul(visible_samples, weights)
                                                      + tf.transpose(hidden_bias)), 1, keep_dims=True))
        return free_energy   

    # make sure signs are correct with free_energy
    # alter so that stochastic_maximum_likelihood doesn't need to be run twice
    # ELN: adding; modified from neg_log_likelihood_grad
    def reverse_kl_div(self, visible_samples, visible_probs, visible_q_probs, alpha=0, num_gibbs=2):

        # we use the frozen weights except in the energy functions being averaged in the positive and negative phases, since the actual gradient only appears there

        # implements importance sampling:
        weighting = tf.div(tf.exp(-self.free_energy(visible_samples)), visible_q_probs)
        weighting_fr = tf.div(tf.exp(-self.free_energy(visible_samples, frozen_params=True)), visible_q_probs)
        # estimator for partition function using samples with importance sampling weighting
        Zhat = tf.reduce_sum(weighting)
        Zhat_fr = tf.reduce_sum(weighting_fr)   # ELN: should be equivalent to: partition_ftn_sample_estimate(self, visible_samples, visible_q_probs, frozen_params=True)

        hidden_samples, _ = self.sample_h_given(visible_samples)
        self.hidden_samples = hidden_samples
        # ELN: this sets hidden_samples = <h>_v for clamped v=visible_samples
        if self.pos_phase_prob==1:
            hidden_samples = tf.sigmoid(tf.transpose(self.hidden_bias) + tf.matmul(visible_samples,self.weights))
            self.hidden_samples = hidden_samples
        energy_pos_phase = self.energy(hidden_samples, visible_samples)

        # this estimator of negative phase uses same samples as positive phase, to remove sample variance in difference
        # note that using frozen weighting and Zhat here has BIG effect in slowing down / stabilizing the learning (do same in neg_log_likelihood??)
        model_visible = visible_samples
        model_hidden = hidden_samples  # use same h's as in positive phase
        energy_neg_phase = tf.reduce_sum(tf.multiply(weighting_fr, self.energy(model_hidden, model_visible)))
        # normalize:
        energy_neg_phase = tf.div(energy_neg_phase, Zhat_fr)

        # IF reverse_kl_div is added to the cost AFTER neg_log_likelihood_grad is used, updating the negative phase samples
        # ... (=NOT ideal) then we can just use those samples without calling stochastic_maximum_likelihood again
        # model_hidden, model_visible = self.hidden_samples, self.visible_samples_model

        # I modified this line to remove the split_samples() optin        
        # model_hidden, model_visible = self.stochastic_maximum_likelihood(num_gibbs,visible_samples)
        #energy_neg_phase = tf.reduce_mean(self.energy(model_hidden, model_visible))

        if alpha==0: kl_weighting = tf.log(visible_probs) + self.free_energy(visible_samples, frozen_params=True)
        else:
            ratio = tf.div(visible_probs, tf.exp(-self.free_energy(visible_samples, frozen_params=True)))
            alph = tf.constant(alpha)
            kl_weighting = tf.pow(ratio, alph)
        kl_weighting = tf.multiply(kl_weighting, weighting_fr)

        # does it interpret energy_neg_phase as a tensor of length = # visible_samples ?
        pos_phase = tf.reduce_sum(tf.multiply(kl_weighting, energy_pos_phase))
        # TEMPORARY CHECK TO CONFIRM COST=0 WITHOUT THE LOG: pos_phase = tf.reduce_sum(tf.multiply(weighting_fr, energy_pos_phase))
        final = tf.div(pos_phase, Zhat_fr) - energy_neg_phase

        return final

    def neg_log_likelihood_grad(self, visible_samples, visible_probs, visible_q_probs, model_samples=None, num_gibbs=2):
        # type: (tf.Tensor, tf.Tensor, int) -> tf.Tensor

        # SAMPLES FOR POSITIVE PHASE
#        hidden_samples, _ = self.sample_h_given(visible_samples)
        # ELN: this sets hidden_samples = <h>_v for clamped v=visible_samples
#        if self.pos_phase_prob==1:
        # SHOULD BE ABLE TO USE self.p_of_h_given() here
#            hidden_samples = tf.sigmoid(tf.transpose(self.hidden_bias) + tf.matmul(visible_samples,self.weights))

        # ELN: adding; only importance weight for the positive phase, since the negative phase relaxes to samples from p_model
#        importance_weights = tf.div(visible_probs, visible_q_probs)
#        expectation_from_data = tf.reduce_mean(tf.multiply(self.energy(hidden_samples, visible_samples), importance_weights))
        # alternative: no importance weighting (q=p):
        # expectation_from_data = tf.reduce_mean(self.energy(hidden_samples, visible_samples))

        # EXACT_POSITIVE_PHASE
        v = tf.cast(tf.constant([[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0,0,0]]), tf.float32)
        # ELN: note the shape:
        p = tf.constant([[0.699300], [0.0001666667], [0.0001666667], [0.0001666667], [0.0001666667], [0.0001666667], [0.0001666667], [0.299700]])
        ph1 = self.p_of_h_given(v, frozen_params=True)
        ph0 = tf.constant([[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]) - ph1
        h0 = tf.zeros(shape=(1, 1))
        h1 = tf.ones(shape=(1, 1))
        energy_v = tf.multiply(self.energy(h0, v), ph0) + tf.multiply(self.energy(h1, v), ph1)
        expectation_from_data = tf.reduce_sum(tf.multiply(energy_v, p))

        ## COPIED FROM REVERSE_KL (FOR NOW) = ALTERNATIVE METHOD FOR NEGATIVE PHASE
        #Zhat = tf.reduce_sum(tf.div(tf.exp(-self.free_energy(visible_samples)), visible_q_probs))
        #model_visible = visible_samples
        #model_hidden = hidden_samples  # use same h's as in positive phase
        #weighting = tf.div(tf.exp(-self.free_energy(visible_samples)), visible_q_probs)
        #energy_neg_phase = tf.reduce_sum(tf.multiply(weighting, self.energy(model_hidden, model_visible)))  # this just averages E(v,h(v)) over v's right?
        ## normalize:
        #energy_neg_phase = tf.div(energy_neg_phase, Zhat)
        #expectation_from_model = energy_neg_phase

#        model_hidden, model_visible = self.split_samples(model_samples) or self.stochastic_maximum_likelihood(num_gibbs,visible_samples)
#        expectation_from_model = tf.reduce_mean(self.energy(model_hidden, model_visible))

        # improve this hack:
        all_visible_states = tf.constant([[1,1,1],[1,1,1],[1,1,0],[1,1,0],[1,0,1],[1,0,1],[0,1,1],[0,1,1],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,0],[0,0,0]])
        all_visible_states = tf.cast(all_visible_states, tf.float32)
        all_hidden_states = tf.constant([[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]])
        all_hidden_states = tf.cast(all_hidden_states, tf.float32)

        # I THINK THAT FROZEN_PARAMS SCREWS THIS UP SOMEHOW, SO I'M TURNING THEM OFF IN MODEL_WEIGHTING AND Z FOR NOW
        # OUTPUT SHOWS THEY'RE NOT SET CORRECTLY!!
        # AND I DON'T THINK THE UPDATE IN main.py WORKS

        # Exact NEGATIVE PHASE
        model_hidden, model_visible = all_hidden_states, all_visible_states
        ## ELN: alternatively, use class variables (for negative phase) which have been externally updated by stochastic_maximum_likelihood        
        ## model_hidden, model_visible = self.hidden_samples, self.visible_samples_model
        model_weighting = tf.exp(-self.energy(model_hidden, model_visible, frozen_params=True))
        numerator = tf.reduce_sum(tf.multiply(self.energy(model_hidden, model_visible), model_weighting))
        Zinv = tf.cast(tf.exp(-self.exact_log_Z(frozen_params=True)), tf.float32)
        expectation_from_model = tf.multiply(numerator, Zinv)
##      self.Zinv = Zinv

        # TEMP
#        self.model_h = model_hidden
#        self.model_v = model_visible
#        self.all_hidden = all_hidden_states
#        self.all_vis = all_visible_states

        self.energies = self.energy(model_hidden, model_visible, frozen_params=False)

        self.pmodel = model_weighting * Zinv

        self.pp = expectation_from_data
        self.np = expectation_from_model

        # ELN: update the frozen parameters when we call this function
        tf.assign(self.weights_frozen, self.weights)
        tf.assign(self.visible_bias_frozen, self.visible_bias)
        tf.assign(self.hidden_bias_frozen, self.hidden_bias)

        return expectation_from_data - expectation_from_model

    def neg_log_likelihood(self, visible_samples, log_Z):
        ''' Compute the average negative log likelihood over a batch of visible samples
            NLL = - <log(p(v))> = + <F> + log(Z) '''  # ELN: changing sign of <F> term
        # ELN: adding - sign
        free_energy = - (tf.matmul(visible_samples, self.visible_bias)
                       - tf.reduce_sum(tf.nn.softplus(tf.matmul(visible_samples, self.weights)
                                                      + tf.transpose(self.hidden_bias)), 1))
        return -tf.reduce_mean( - free_energy - log_Z)


    # ELN: this matches exact_log_partition_function() up to error ~1e-5
    def exact_log_Z(self, frozen_params=False):

        # would be nice to generalize this for num_visible>3
        all_visible_states = tf.constant([[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
        all_visible_states = tf.cast(all_visible_states, tf.float32)

        free_energy = self.free_energy(all_visible_states, frozen_params=frozen_params)

        return tf.log(tf.reduce_sum(tf.exp(-free_energy)))


    # ELN: this matches exact_log_partition_function() up to tiny error
    def exact_log_Z_2(self, frozen_params=False):

        # improve this hack, generalize to num_visible>3
        all_visible_states = tf.constant([[1,1,1],[1,1,1],[1,1,0],[1,1,0],[1,0,1],[1,0,1],[0,1,1],[0,1,1],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,0],[0,0,0]])
        all_visible_states = tf.cast(all_visible_states, tf.float32)
        all_hidden_states = tf.constant([[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]])
        all_hidden_states = tf.cast(all_hidden_states, tf.float32)

        p_tilde_joint = tf.exp(-self.energy(all_hidden_states, all_visible_states, frozen_params=frozen_params))

        return tf.log(tf.reduce_sum(p_tilde_joint))


#     # ELN: adding
#     def exact_mean_energy(self):
#         '''Evaluate the mean energy <E(v,h)> of the joint distribution by exact enumerations '''

#         ...

#         log_Z = tf.reduce_sum(tf.exp(-energy))  # equivalent to ______
#         energy_mean_unnorm = tf.reduce_sum(tf.multiply(energy, tf.exp(-energy)))
#         return tf.div(energy_mean_unnorm, log_Z)


# NOTES FOR A METHOD TO RETURN THE NEGATIVE PHASE
#         (self, frozen_params=False):
#         visible_samples
#         weighting = tf.div(tf.exp(-self.free_energy(visible_samples, frozen_params=frozen_params)), visible_q_probs)
#         Zhat = tf.reduce_sum(weighting)
#         model_visible = visible_samples
#         model_hidden = self.hidden_samples
#         # this is currently set to NEVER use frozen parameters in energy()
#         energy_neg_phase = tf.reduce_sum(tf.multiply(weighting, self.energy(model_hidden, model_visible)))
#         energy_neg_phase = tf.div(energy_neg_phase, Zhat)
#         return energy_neg_phase


    def exact_log_partition_function(self):
        ''' Evaluate the partition function by exact enumerations '''
        with tf.name_scope('exact_log_Z'):
            # Define the exponent: H*b + sum(softplus(1 + exp(a + w*H.T)))
            first_term = tf.matmul(self.all_hidden_states, self.hidden_bias, name='first_term')
            with tf.name_scope('second_term'):
                second_term = tf.matmul(self.weights, self.all_hidden_states, transpose_b=True)
                second_term = tf.nn.softplus(tf.add(self.visible_bias, second_term))
                second_term = tf.transpose(tf.reduce_sum(second_term, reduction_indices=[0], keep_dims=True))
            exponent = tf.cast(first_term + second_term, dtype=tf.float64, name='exponent')
            #exponent_mean = tf.reduce_mean(exponent)
            exponent_mean = tf.reduce_max(exponent)

###            self.first_term = first_term

            return tf.log(tf.reduce_sum(tf.exp(exponent - exponent_mean))) + exponent_mean

    # ELN: adding -- THIS DOES NOT CURRENTLY SEEM TO BE WORKING
    def log_partition_ftn_sample_estimate(self, visible_samples, visible_q_probs, frozen_params=True):

        n_samples = tf.cast(tf.size(visible_q_probs), tf.float32)  # checked: this outputs batch size parameter from main.py
        weighting = tf.div(tf.exp(-self.free_energy(visible_samples, frozen_params=frozen_params)), visible_q_probs)
        ## two = tf.constant(2.0)  # trying to express "2^n" in tf objects
        ## normalize = tf.div(tf.pow(two, self.num_visible), n_samples)
        return tf.log(tf.div(tf.reduce_sum(weighting), n_samples))

    # ELN: adding; NOT currently in use; I DON'T THINK THIS WORKS; is use of hidden_samples correct??
    def negative_phase_sample_estimate(self, visible_samples, visible_q_probs, frozen_params=True):

        weighting = tf.div(tf.exp(-self.free_energy(visible_samples, frozen_params=frozen_params)), visible_q_probs)
        Zhat = tf.reduce_sum(weighting)

        model_visible = visible_samples
        model_hidden = self.hidden_samples
        # this is currently set to NEVER use frozen parameters in energy()
        energy_neg_phase = tf.reduce_sum(tf.multiply(weighting, self.energy(model_hidden, model_visible)))
        energy_neg_phase = tf.div(energy_neg_phase, Zhat)

        return energy_neg_phase

    # ELN: adding, modified from exact_log_partition_function() above; self.all_hidden_state -> self.hidden_states (added as argument)
    # ELN: double check that dividing by Z correctly gives marginal probabilities
    def unnorm_log_p_h_marginal(self, hidden_states):
        ''' Evaluate the marginal log-probability of a configuration of hidden units '''
        with tf.name_scope('exact_log_p_h'):  # ELN: need to look up tf.name_scope()
            # Define the exponent: H*b + sum(softplus(1 + exp(a + w*H.T)))
            first_term = tf.matmul(hidden_states, self.hidden_bias, name='first_term')
            with tf.name_scope('second_term'):
                second_term = tf.matmul(self.weights, hidden_states, transpose_b=True)
                second_term = tf.nn.softplus(tf.add(self.visible_bias, second_term))
                second_term = tf.transpose(tf.reduce_sum(second_term, reduction_indices=[0], keep_dims=True))
            exponent = tf.cast(first_term + second_term, dtype=tf.float64, name='exponent')

            return exponent

    def split_samples(self, samples):
        if samples is None:
            return None

        visible_samples = tf.slice(samples, begin=(0, 0), size=(self.num_samples, self.num_visible))
        hidden_samples = tf.slice(samples, begin=(0, self.num_visible), size=(self.num_samples, self.num_hidden))
        return hidden_samples, visible_samples

    @staticmethod
    def sample_binary_tensor(prob, m, n):
        # type: (tf.Tensor, int, int) -> tf.Tensor
        """
        Convenience method for generating a binary Tensor using a given probability.

        :param prob: Tensor of shape (m, n)
        :param m: number of rows in result.
        :param n: number of columns in result.
        """
        return tf.where(
            tf.less(tf.random_uniform(shape=(m, n)), prob),
            tf.ones(shape=(m, n)),
            tf.zeros(shape=(m, n))
        )
