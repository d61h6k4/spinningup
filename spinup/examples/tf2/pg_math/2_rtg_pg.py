import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gymnasium as gym
from gymnasium.spaces import Discrete, Box


def mlp(sizes, activation=tf.keras.activations.tanh, output_activation=tf.keras.layers.Lambda(lambda x: x)):
    inputs = tf.keras.Input(shape=sizes[0])
    x = tf.keras.Sequential(layers=[tf.keras.layers.Dense(size, activation=activation)
                                    for size in sizes[1:]])(inputs)
    outputs = output_activation(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    print(f"{obs_dim=}, {n_acts=}")
    # make core of policy network
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts], activation=tf.keras.activations.gelu)

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return tfp.distributions.Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().numpy()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -tf.reduce_mean(logp * weights)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs, _ = env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(tf.convert_to_tensor([obs], dtype=tf.float32))[0]
            obs, rew, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, _ = env.reset()
                done, ep_rews = False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        with tf.GradientTape() as tape:
            batch_loss = compute_loss(obs=tf.convert_to_tensor(batch_obs, dtype=tf.float32),
                                      act=tf.convert_to_tensor(batch_acts, dtype=tf.int32),
                                      weights=tf.convert_to_tensor(batch_weights, dtype=tf.float32)
                                      )
        grads = tape.gradient(batch_loss, logits_net.trainable_weights)
        optimizer.apply_gradients(zip(grads, logits_net.trainable_weights))

        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print(
            f"epoch: {i:d} \t loss: {batch_loss:.3f} \t return: {np.mean(batch_rets):.3f} \t ep_len: {np.mean(batch_lens):.3f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
