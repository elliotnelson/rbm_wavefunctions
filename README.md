# rbm_wavefunctions

This repository contains a TensorFlow implementation of Restricted Boltzmann Machine (RBM) learning, adapted from [github.com/MelkoCollective/ICTP-SAIFR-MLPhysics/tree/master/RBM_CDL](https://github.com/MelkoCollective/ICTP-SAIFR-MLPhysics/tree/master/RBM_CDL).

The current goal is to train the RBM to efficiently represent a simple simple multi-modal data distribution.  In particular, I am exploring the success of different cost functions at maintaining probability mass over multiple modes while also suppressing probability mass between modes, where probability can be extremely low.

This is early-stage work in progress with collaborators, and the goal is to extend results for multi-modal probability distributions to entangled quantum wavefunctions with a similar structure.  (Efficient learning of multi-modal structure with a polynomial number of hidden nodes could ultimately allow for a speedup of quantum simulations of systems with an exponentially large number of (visible node) configurations.)
