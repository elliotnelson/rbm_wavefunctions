# RBM Learning of Multi-Modal Distributions

This repository contains a TensorFlow implementation of Restricted Boltzmann Machine (RBM) learning, adapted from [github.com/MelkoCollective/ICTP-SAIFR-MLPhysics/tree/master/RBM_CDL](https://github.com/MelkoCollective/ICTP-SAIFR-MLPhysics/tree/master/RBM_CDL).

The current goal is to train the RBM to efficiently represent a simple simple multi-modal data distribution.  In particular, I am exploring the success of different cost functions at maintaining probability mass over multiple modes while also suppressing probability mass between modes, where probability can be extremely low.

In general, unsupervised data are drawn from some unknown probability distribution, and the cross-entropy or KL divergence between the true distribution *p(x)* and model distribution <a href="https://www.codecogs.com/eqnedit.php?latex=p_M(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_M(x)" title="p_M(x)" /></a> must be approximated as a sum over the data samples available:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{L}&space;=&space;\sum_x&space;p(x)\log&space;p_M(x)&space;\approx&space;\frac{1}{N}\sum_i^N&space;\log&space;p_M(x_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{L}&space;=&space;\sum_x&space;p(x)\log&space;p_M(x)&space;\approx&space;\frac{1}{N}\sum_i^N&space;\log&space;p_M(x_i)" title="\mathcal{L} = \sum_x p(x)\log p_M(x) \approx \frac{1}{N}\sum_i^N \log p_M(x_i)" /></a>

However, if we are able to compute the true probability *p(x)* for a sample configuration *x*, we can use importance sampling and train on samples <a href="https://www.codecogs.com/eqnedit.php?latex=x_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_j" title="x_j" /></a> drawn from a simpler distribution *q(x)*, weighting each by *p(x)/q(x)* when computing the cross-entropy:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{L}&space;\approx&space;\frac{1}{N}\sum_j^N&space;\frac{p(x_j)}{q(x_j)}&space;\log&space;p_M(x_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{L}&space;\approx&space;\frac{1}{N}\sum_j^N&space;\frac{p(x_j)}{q(x_j)}&space;\log&space;p_M(x_j)" title="\mathcal{L} \approx \frac{1}{N}\sum_j^N \frac{p(x_j)}{q(x_j)} \log p_M(x_j)" /></a>

This also allows us to use other cost functions which do not integrate over the true distribution, such as the reverse KL divergence, which integrates over the (RBM) model distribution.

In particular, the reverse KL divergence (unlike the usual cross-entropy or KL divergence) strongly penalizes the model for failing to assign low probability to unlikely configurations with low *p(x)*. Using an importance sampling distribution *q(x)* to include such configurations, along with a cost function which penalizes in this way, we can more easily train the RBM to suppress probability mass between the modes of *p(x)*.  On the other hand, the reverse KL divergence has a tendency to lock onto a single mode, and struggles to maintain probability mass over all modes as well as the cross-entropy.  (Cf. Goodfellow, ["NIPS 2016 Tutorial:
Generative Adversarial Networks"](https://arxiv.org/pdf/1701.00160.pdf).)  I am currently looking for a good way to resolve this.

This is early-stage work in progress with collaborators, and the goal is to extend results for multi-modal probability distributions to entangled quantum wavefunctions with a similar structure.  (Efficient learning of multi-modal structure with a polynomial number of hidden nodes could ultimately allow for a speedup of quantum simulations of systems with an exponentially large number of (visible node) configurations.)
