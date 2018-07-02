from __future__ import print_function
from pprint import pformat
import tensorflow as tf
import itertools as it
from random import randint
import argparse
from rbm import RBM
import numpy as np
import json
import math

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('command',type=str,help='command to execute') 
    parser.add_argument('-nV',type=int,default=3,help='number of visible nodes')  #  ideally, read this off from dataset
    parser.add_argument('-nH',type=int,default=1,help='number of hidden nodes')
    parser.add_argument('-steps',type=int,default=10000,help='training steps')
    parser.add_argument('-lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('-bs',type=int,default=100,help='batch size')
    parser.add_argument('-CD',type=int,default=3,help='initial steps of contrastive divergence')
    parser.add_argument('-pos_phase_prob',type=int,default=0,help='Boolean for taking mean <vh>=v<h>_v in positive phase with sigmoid')
    parser.add_argument('-PCD_prob',type=int,default=0,help='Boolean for taking mean <vh>=<h<v>_h> in negative phase with sigmoid')
    parser.add_argument('-not_PCD',type=int,default=0,help='Boolean for initializing CD chains with data instead of persistent configurations')
    parser.add_argument('-nC',type=float,default=100,help='number of chains in PCD')  # the number of "fantasy particles" in PCD = Persistent Contrastive Divergence, works best with small learning rates; note that this gets fed to the rbm class instantiation as num_samples, and is then used to define rbm.hidden_samples, which is used in rbm.stochastic_maximum_likelihood(), which is used in rbm.neg_log_likelihood_grad()
    parser.add_argument('-beta1', type=float,default=0*0.05,help='l1 regularization coefficient')
    parser.add_argument('-beta2', type=float,default=0*0.001,help='l2 regularization coefficient')
    parser.add_argument('-alpha', type=float,default=0.9,help='momentum parameter')
    parser.add_argument('-rmsprop_decay', type=float,default=0.8,help='RMSProp decay parameter')
    parser.add_argument('-decay', type=float,default=0.1,help='learning decay rate')    # 1 = no decay; this is TOTAL decay over nsteps
    parser.add_argument('-epoch_size', type=float,default=100,help='epoch size')    # defined to be the number of iterations per epoch
    parser.add_argument('-alpha_renyi', type=float,default=0,help='parameter specifying cost function within family of Renyi divergences')   # note that 0 fixes to reverse KL divergence

    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    
    if args.command == 'sample':
        sample(args)


class Args(object):
    pass

class Placeholders(object):
    pass

class Ops(object):
    pass

def train(args):
   
    # Simulation parameters
    num_visible = args.nV           # number of visible nodes
    num_hidden = args.nH            # number of hidden nodes
    nsteps = args.steps             # training steps
    bsize = args.bs                 # batch size
    learning_rate_b=args.lr         # learning rate
    num_gibbs = args.CD             # number of Gibbs iterations
    num_samples = args.nC           # number of chains in PCD
    weights=None                    # weights
    visible_bias=None               # visible bias
    hidden_bias=None                # hidden bias
    bcount=0                        # counter
    epochs_done=1                   # epochs counter
    # ELN: additional parameters:
    beta1 = args.beta1              # l1 regularization coefficient
    beta2 = args.beta2              # l2 regularization coefficient
    alpha = args.alpha              # momentum parameter
    rmsprop_decay = args.rmsprop_decay
    decay_rate = args.decay
    PCD_prob = args.PCD_prob
    pos_phase_prob = args.pos_phase_prob
    not_PCD = args.not_PCD
    alpha_renyi = args.alpha_renyi

    # *************************************************************
    # INSERT HERE THE PATH TO THE TRAINING AND TESTING DATASETS
    x_trainName = 'x_ghz.txt'
    p_trainName = 'prob_ghz.txt'
    q_trainName = 'q_prob_ghz.txt'
    # testName = ...    # optional: fill in testing dataset

    # Loading the data
    xtrain = np.loadtxt(x_trainName)
    p_labels = np.loadtxt(p_trainName)  # labels p(x) for the true probability of data x
    # optional; for importance sampling:
    q_labels = np.loadtxt(q_trainName)  # labels q(x) for the probability of the distribution used for importance sampling

    # xtest = np.loadtxt(testName)     # optional: load test data
    
    ## DON'T SHUFFLE UNLESS LABELS RESHUFFLED IN SAME WAY
    # ept=np.random.permutation(xtrain)               # random permutation of training data
    # epv=np.random.permutation(xtest)                # random permutation of test data
    iterations_per_epoch = args.epoch_size  # xtrain.shape[0] / bsize  # gradient iteration per epoch

    # Initialize RBM class
    rbm = RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples)

    ## incorpororate all these + added __init__ vars (specify args) into args argument?
    rbm.gibbs_chains_end_prob = PCD_prob  ## explanation
    rbm.pos_phase_prob = pos_phase_prob
    rbm.not_pcd = not_PCD

    # Initialize operations and placeholders classes
    ops = Ops()
    placeholders = Placeholders()

    placeholders.visible_samples = tf.placeholder(tf.float32, [None, num_visible], name='v') # placeholder for training data
    placeholders.visible_p = tf.placeholder(tf.float32, [None], name='p') # ELN: adding
    placeholders.visible_q = tf.placeholder(tf.float32, [None], name='q') # ELN: adding
    ## shape=(None, num_visible)

    total_iterations = 0 # starts at zero
    ops.global_step = tf.Variable(total_iterations, name='global_step_count', trainable=False)
    
    # Decaying learning rate
    learning_rate = tf.train.exponential_decay(
        learning_rate_b,
        ops.global_step,
        nsteps, # ELN: default was: 100 * xtrain.shape[0]/bsize,
        decay_rate # ELN: default was 1 ... # decay rate =1 means no decay
    )

    # ELN: adding L1 loss
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=1.0, scope=None)
    #    tf.get_collection("weights_list")
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, rbm.weights)  # see TensorFlow -> programmer's guide -> variables
    ## kl = tf.Variable(0.)
    cost = rbm.neg_log_likelihood_grad(placeholders.visible_samples, placeholders.visible_p, placeholders.visible_q, num_gibbs=num_gibbs)
    cost_kl = 0.00005 * rbm.reverse_kl_div(placeholders.visible_samples, placeholders.visible_p, placeholders.visible_q, alpha=alpha_renyi, num_gibbs=num_gibbs)
    cost_reg = beta1 * tf.contrib.layers.apply_regularization(l1_regularizer) + beta2 * tf.nn.l2_loss(rbm.weights)/(num_visible*num_hidden) # ELN: adding L2 regularization
    # cost += cost_reg
    # cost_kl += cost_reg

    #### PROBABLY BOTH WRONG!!:
    #logZhat = rbm.log_partition_ftn_sample_estimate(placeholders.visible_samples, placeholders.visible_q)
    #np_hat = rbm.negative_phase_sample_estimate(placeholders.visible_samples, placeholders.visible_q)

    optimizer = tf.train.MomentumOptimizer(learning_rate, alpha)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=rmsprop_decay, momentum=alpha)
    # optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-5)

    # Define operations
    ops.lr=learning_rate
    vars_train = [rbm.weights, rbm.visible_bias, rbm.hidden_bias]  # ELN: adding this to prevent updating of "frozen" parameters
    ops.train = optimizer.minimize(cost, global_step=ops.global_step, var_list=vars_train)  # ELN: increases ops.global_step by 1 upon minimizing cost
    ops.train_kl = optimizer.minimize(cost_kl, global_step=ops.global_step, var_list=vars_train)
    ops.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # ELN: for comparing P(0's) to P(1's)
    hidden_zeros = tf.zeros(shape=(1, num_hidden))
    hidden_ones = tf.ones(shape=(1, num_hidden))
 
    with tf.Session() as sess:
        sess.run(ops.init)

        # ELN: print the initial weights
        weights = sess.run(rbm.weights)
        print('Initial Weights')
        print(weights)

        # ELN: define the mean square of weights
        # weights_0 = sess.run(rbm.weights)
        # weights_mean_square_0 = sess.run(tf.reduce_mean(tf.square(weights_0)))

        for ii in range(nsteps):  # ELN: updates parameters nsteps times, going through training data nsteps/xtrain.shape[0] times
            if bcount*bsize+ bsize>=xtrain.shape[0]:  # ELN: counts what bin of training data the current batch is; True until we go through the training set
               bcount=0   # ELN: resets batch counter to 0 after we go through training set
#               ept=np.random.permutation(xtrain)  # ELN: re-shuffles the training data after each pass through training data

            batch = xtrain[bcount*bsize: bcount*bsize + bsize]
            batch_ps = p_labels[bcount*bsize: bcount*bsize + bsize]
            batch_qs = p_labels[bcount*bsize: bcount*bsize + bsize]

            bcount=bcount+1  # ELN: gets updates until hitting bcount_max = n_batches = xtrain.shape[0]/bsize - 1 (see above), then starts over
            feed_dict = {placeholders.visible_samples: batch, placeholders.visible_p: batch_ps, placeholders.visible_q: batch_qs}  # ELN: dictionary to pass the current batch into visible_samples to train rbm
            # feed_dict_x = {placeholders.visible_samples: batch}  # ELN: this is only for the data, no labels

            ### ELN: update the negative phase (v,h) samples; a dictionary is only needed if we used CD not PCD
            ### ELN: for some reason this takes way too long when run outside of neg_log_likelihood_grad, so I'm currently just running it there and subsequently using the updated negative phase samples in reverse_kl_div without running it again:
            ### _, _ = sess.run(rbm.stochastic_maximum_likelihood(num_gibbs))

            # ELN: option for switching cost functions:
            # epochs_switch = 200
            # if epochs_done<epochs_switch:
            #    _, num_steps, loss = sess.run([ops.train, ops.global_step, cost], feed_dict=feed_dict)  # trains until ops.global_step is increased by 1 by ops.train, and then fixes num_steps to updated global_step; feed_dict feeds the current training batch into placeholders.visible_samples
            # else:
            
            ##num_steps = bcount
            # _, num_steps, loss, logZ_est, np_est = sess.run([ops.train_kl, ops.global_step, cost_kl, logZhat, np_hat], feed_dict=feed_dict)
            _, num_steps, loss = sess.run([ops.train, ops.global_step, cost], feed_dict=feed_dict)  # trains until ops.global_step is increased by 1 by ops.train, and then fixes num_steps to updated global_step; feed_dict feeds the current training batch into placeholders.visible_samples

            # ELN: this updates the frozen = non-trainable parameters so they can be used on next iteration
            # IS THIS RIGHT? did I decide it is??
            #rbm.weights_frozen, rbm.hidden_bias_frozen, rbm.visible_bias_frozen = sess.run([rbm.weights, rbm.hidden_bias, rbm.visible_bias])
            ## DOESN'T WORK:
            ##tf.assign(rbm.weights_frozen, sess.run(rbm.weights))
            ##tf.assign(rbm.hidden_bias_frozen, sess.run(rbm.hidden_bias))
            ##tf.assign(rbm.visible_bias_frozen, sess.run(rbm.visible_bias))

            # ELN: num_steps is basically the same as ii (?), running up to nsteps

            if 0==0: #num_steps % iterations_per_epoch == 0:
                print ('Epoch = %d     ' % epochs_done)
                print ('Learning rate = %f ' % sess.run(learning_rate))
                ## ELN: the current ratio of probabilities for 0's to 1's
                ## ELN: this is just zero: # energy_0 = sess.run(rbm.energy(hidden_zeros, visible_zeros))
                ## energy_1 = sess.run(rbm.energy(hidden_ones, visible_ones))
                ## prob_other = math.exp(-2*logZ) * math.exp(-(0+energy_1))
                #Z1 = sess.run(tf.exp(rbm.exact_log_partition_function()))
                #Z2 = sess.run(tf.exp(rbm.exact_log_Z()))
                p_v1_given_h0 = sess.run(rbm._of_v_given(hidden_zeros))
                p_v1_given_h1 = sess.run(rbm.p_of_v_given(hidden_ones))
                p_v0_given_h0 = 1. - p_v1_given_h0
                p_v0_given_h1 = 1. - p_v1_given_h1
                p000_given_h0 = np.prod(p_v0_given_h0)
                p111_given_h0 = np.prod(p_v1_given_h0)
                p000_given_h1 = np.prod(p_v0_given_h1)
                p111_given_h1 = np.prod(p_v1_given_h1)
                # ELN: the following two lines can slow down computation after a while...
                ph0_tilde = sess.run(tf.exp(rbm.unnorm_log_p_h_marginal(hidden_zeros)))  # ELN: normalize by dividing by partition function
                ph1_tilde = sess.run(tf.exp(rbm.unnorm_log_p_h_marginal(hidden_ones)))  # ELN: normalize by dividing by partition function
                ph0 = ph0_tilde / (ph0_tilde+ph1_tilde)
                ph1 = 1. - ph0
                print('rbm.pp = ')
#                print(sess.run(rbm.pp))
#                print('rbm.np = ')
#                print(sess.run(rbm.np))
#                print('rbm.energies = ')
#                print(sess.run(rbm.energies))
#                print('rbm.pmodel = ')
#                print(sess.run(rbm.pmodel))
##                print('sum of p model should be 1: ')
##                print(sess.run(tf.reduce_sum(rbm.pmodel)))
##                print('rbm.Zinv = ')
##                print(sess.run(rbm.Zinv))
                print('Probabilities of v_i=0 given h=0:')
                print(p_v0_given_h0)
                print('Probabilities of v_i=1 given h=1:')
                print(p_v1_given_h1)
                print('p(h=0):')
                print(ph0)
                print('p(h=1):')
                print(ph1)
                print('p(v=000):')
                print(ph0*p000_given_h0 + ph1*p000_given_h1)
                print('p(v=111):')
                print(ph1*p111_given_h1 + ph0*p111_given_h0)
                # print ('Energy of 1s = %f' % energy_1)
                # print ('Probability of NON-GHZ configuration = %f' % prob_other)
                # print ('Energy of 0s with 1 visible 1 = %f' % energy_one1)
                weights, v_bias, h_bias, weights_fr = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias, rbm.weights_frozen])  # ELN: adding from save_parameters
# ELN: cost = wrong dtype in this line:                print('COST = %d' % cost)  # ELN: adding
                print('Loss')  # ELN: adding
                print(loss)  # ELN: adding                
                #print('logZ-hat')  # ELN: adding
                #print(logZ_est)  # ELN: adding                
                ##print('Negative phase <E> sample estimate')  # ELN: adding
                ##print(np_est)  # ELN: adding                
                print('Weights frozen')  # ELN: adding
                print(weights_fr)  # ELN: adding
                print('Weights')  # ELN: adding
                print(weights)  # ELN: adding
                print('Visible Bias')  # ELN: adding
                print(v_bias)  # ELN: adding
                print('Hidden Bias')  # ELN: adding
                print(h_bias)  # ELN: adding
                epochs_done += 1
                # ELN: adding the following to increase # of CD steps as weights increase
                # weights_mean_square = sess.run(tf.reduce_mean(tf.square(weights)))
                # if weights_mean_square>2*weights_mean_square_0:
                #    num_gibbs = 2*num_gibbs
                #    weights_mean_square_0 = sess.run(tf.reduce_mean(tf.square(weights)))
                #    print('NUMBER OF STEPS IN CD = %d   ' % num_gibbs)

                save_parameters(sess, rbm, epochs_done)  # ELN: modified, moved here (indented)

def sample(args):
       
    num_visible = args.nV   # number of visible nodes
    num_hidden = args.nH    # number of hidden nodes
    
    # *************************************************************
    # INSERT HERE THE PATH TO THE PARAMETERS FILE
    path_to_params = 'ghz_w.txt.npz'  # ELN: modified
   
    # Load the RBM parameters 
    params = np.load(path_to_params)
    weights = params['weights']
    visible_bias = params['visible_bias']
    hidden_bias = params['hidden_bias']
    hidden_bias=np.reshape(hidden_bias,(hidden_bias.shape[0],1))
    visible_bias=np.reshape(visible_bias,(visible_bias.shape[0],1))
  
    # Sampling parameters
    num_samples=1   # how many independent chains will be sampled
                    # ELN: modified
    gibb_updates=20  # how many gibbs updates per call to the gibbs sampler
                     # ELN: default=100 
    nbins=40         # number of calls to the RBM sampler      
                     # ELN: modified

    # Initialize RBM class
    # ELN: note that rbm.hidden_samples is (randomly) instantiated when RBM is called
    rbm = RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples)
    hsamples,vsamples=rbm.stochastic_maximum_likelihood(gibb_updates)

    # ELN: you can break the PCD chain to get uncorrelated sample, but you can't just draw from desired p(h) marginal distribution by hand, so should we use something like tempering in order to sample? (references in Goodfellow book)

    # Initialize tensorflow
    init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
   
    with tf.Session() as sess:
        sess.run(init)

        print('Samples: ')  # ELN: adding
    
        for i in range(nbins):
            # print ('bin %d\t' %i)
            
            # Gibbs sampling
            _,samples=sess.run([hsamples,vsamples])
            print(samples)     # ELN: adding

            # ELN: added from RBM's __init__ method, to re-initialize rbm.hidden_samples and then burn in the CD chain for each sample
            rbm.hidden_samples = tf.Variable(
                rbm.sample_binary_tensor(tf.constant(0.5), rbm.num_samples, rbm.num_hidden),
                trainable=False, name='hidden_samples')


def save_parameters(sess,rbm,epochs):
    weights, visible_bias, hidden_bias = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias])


    # *************************************************************
    # INSERT HERE THE PATH TO THE PARAMETERS FILE
    parameter_file_path = 'ghz_w.txt'  # ELN: modified
    
    np.savez_compressed(parameter_file_path, weights=weights, visible_bias=visible_bias, hidden_bias=hidden_bias,
                        epochs=epochs)


if __name__=='__main__':
    main()
