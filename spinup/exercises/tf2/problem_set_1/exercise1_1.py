import tensorflow as tf
import numpy as np

"""

Exercise 1.1: Diagonal Gaussian Likelihood

Write a function which takes in Tensorflow symbols for the means and
log stds of a batch of diagonal Gaussian distributions, along with a
Tensorflow placeholder for (previously-generated) samples from those
distributions, and returns a Tensorflow symbol for computing the log
likelihoods of those samples.

"""

EPS = 1e-8


def gaussian_likelihood(x, mu, log_std):
    """
    Args:
        x: Tensor with shape [batch, dim]
        mu: Tensor with shape [batch, dim]
        log_std: Tensor with shape [batch, dim] or [dim]

    Returns:
        Tensor with shape [batch]
    """
    k = tf.cast(tf.shape(mu)[1], tf.float32)

    sigma = tf.exp(log_std)
    sigma2 = sigma ** 2.0 + EPS

    return -0.5 * (tf.reduce_sum((x - mu) ** 2.0 / sigma2 + 2.0 * log_std, axis=-1) + k * tf.math.log(2.0 * np.pi))


if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """
    from spinup.exercises.tf2.problem_set_1_solutions import exercise1_1_soln
    from spinup.exercises.common import print_result

    dim = 10
    batch_size = 32
    x = tf.convert_to_tensor(np.random.rand(batch_size, dim), dtype=tf.float32)
    mu = tf.convert_to_tensor(np.random.rand(batch_size, dim), dtype=tf.float32)
    log_std = tf.convert_to_tensor(np.random.rand(dim), dtype=tf.float32)

    your_result = gaussian_likelihood(x, mu, log_std)
    true_result = exercise1_1_soln.gaussian_likelihood(x, mu, log_std)

    correct = np.allclose(your_result, true_result)
    print_result(correct)
