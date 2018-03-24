# Policy Gradient
# environment: https://gym.openai.com/envs/Pong-v0/
# 2 actions: up and down
# inspired by https://github.com/mrahtz/tensorflow-rl-pong
# Based on Andrej Karpathy's "Deep Reinforcement Learning: Pong from Pixels" http://karpathy.github.io/2016/05/31/rl/

import numpy as np
import gym
from policyGradientNetwork import Network

# variables
hidden_layer_size=200
learning_rate=0.0006
checkpointNumber=10
load_checkpoint=True
discount_factor=0.99
render=True
batch_size = 1

# preprocess
def preprocess(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

# compute discounted rewards
def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(rewards)):
            discounted_reward_sum += rewards[k] * discount
            discount *= discount_factor
            if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards

if __name__ == '__main__':
    # use gym environment: Pong-v0
    # initialization of environment
    environment = gym.make("Pong-v0")
    observation = environment.reset()

    # Action values to send to gym environment to move paddle up/down
    UP_ACTION = 2
    DOWN_ACTION = 3
    action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

    policyTFNetwork = Network(hidden_layer_size, learning_rate, checkpoints_dir='checkpoints')
    if load_checkpoint:
        policyTFNetwork.load_checkpoint()

    batch_state_action_reward_tuples = []
    smoothed_reward = None
    episode_number = 1

    while True:
        done = False
        total_reward = 0
        round_number = 1

        last_observation = environment.reset()
        last_observation = preprocess(last_observation)
        action = environment.action_space.sample()

        # step the environment and get new measurements
        observation, reward, done, info = environment.step(action)
        observation = preprocess(observation)
        number_steps = 1

        while not done:
            if render:
                environment.render()

            observation_delta = observation - last_observation
            last_observation = observation
            update_probability = policyTFNetwork.forward_pass(observation_delta)[0]
            if np.random.uniform() < update_probability:
                action = UP_ACTION
            else:
                action = DOWN_ACTION

            # step the environment and get measurements
            observation, reward, done, info = environment.step(action)
            observation = preprocess(observation)
            total_reward += reward
            number_steps += 1

            # add action, observation_delta and reward
            tup = (observation_delta, action_dict[action], reward)
            batch_state_action_reward_tuples.append(tup)

            if reward != 0:
                print('Episode no: %d,  Game round finished, Reward: %f, Total Reward: %f' % (episode_number, reward, total_reward))
                round_number += 1
                n_steps = 0

        # exponentially smoothed version of reward
        if smoothed_reward is None:
            smoothed_reward = total_reward
        else:
            smoothed_reward = smoothed_reward * 0.99 + total_reward * 0.01
        print("Total Reward was %f; running mean reward is %f" % (total_reward, smoothed_reward))

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            states, actions, rewards = zip(*batch_state_action_reward_tuples)
            rewards = discount_rewards(rewards, discount_factor)
            rewards = rewards - np.mean(rewards)
            rewards = rewards / np.std(rewards)
            batch_state_action_reward_tuples = list(zip(states, actions, rewards))
            policyTFNetwork.train(batch_state_action_reward_tuples)
            batch_state_action_reward_tuples = []

        # save episodes for checkpoint
        if episode_number % checkpointNumber == 0:
            policyTFNetwork.save_checkpoint()

        episode_number += 1

