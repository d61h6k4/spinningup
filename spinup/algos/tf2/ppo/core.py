import scipy.signal

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gymnasium.spaces import Box, Discrete


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.trainable_weights])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp(sizes, activation=tf.keras.activations.tanh, output_activation=tf.keras.layers.Lambda(lambda x: x)):
    inputs = tf.keras.Input(shape=sizes[0])
    x = tf.keras.Sequential(layers=[tf.keras.layers.Dense(size, activation=activation)
                                    for size in sizes[1:]])(inputs)
    outputs = output_activation(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


class Actor(tf.keras.Model):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def call(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return tfp.distributions.Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * tf.ones(act_dim, dtype=tf.float32)
        self.log_std = tf.Variable(initial_value=log_std, trainable=True, name="log_std")
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = tf.exp(self.log_std)
        return tfp.distributions.Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return tf.reduce_sum(pi.log_prob(act), axis=-1)


class MLPCritic(tf.keras.Model):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self._v = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def call(self, obs):
        return tf.squeeze(self._v(obs), axis=-1)


class MLPActorCritic(tf.keras.Model):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=tf.keras.activations.tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        pi = self.pi._distribution(obs)
        a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.pi._distribution(obs).sample()
