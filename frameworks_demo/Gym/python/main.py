from __future__ import division
from __future__ import absolute_import

import gym
import tensorflow as tf
import numpy as np
from dqn import dqn
from replay_buffer import replay_buffer

# Parameters
BATCH_SIZE = 32
NUM_EPISODES = 100000
MIN_SAMPLES = 2 * BATCH_SIZE  # Minimal number of samples for an update
EVAL_EPISODE = 10  # Perform evaluation each EVAL_EPISODE number of episodes
NUM_STEPS = 200
NUM_EVALUATIONS = 10  # Number of times we run evaluation
INITIAL_EPSILON = 0.9
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.999
BUFFER_SIZE = 100000

env = gym.make('CartPole-v0')

ndim_action = env.action_space.n
ndim_obs = env.observation_space.shape[0]
epsilon = INITIAL_EPSILON

agent = dqn(ndim_obs, ndim_action)
replay = replay_buffer(BUFFER_SIZE)

for i in range(NUM_EPISODES):
    # Training
    obs = env.reset()
    for j in range(NUM_STEPS):
        # Epsilon greedy exploration
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.act(np.expand_dims(obs, 0))[0]

        # Perform an action, observe next state and reward
        newobs, reward, done, info = env.step(action)

        # Insert it to replay buffer
        replay.insert(obs, action, reward, newobs, 0 if done == True else 1)

        if done == True:
            break
        else:
            obs = newobs

        if len(replay.deque) >= MIN_SAMPLES:
            # Decay epsilon
            epsilon = epsilon * EPSILON_DECAY
            epsilon = max(epsilon, MIN_EPSILON)

            # Sample a batch of samples and then update
            obserbation_batch, action_batch, reward_batch, next_obserbation_batch, mask_batch = replay.sample(
                BATCH_SIZE)
            agent.update(obserbation_batch, action_batch, reward_batch,
                         next_obserbation_batch, mask_batch)

    agent.save()

    # Evaluation
    if i % EVAL_EPISODE == 0:
        total_reward = 0
        for k in range(NUM_EVALUATIONS):
            obs = env.reset()
            for j in range(NUM_STEPS):
                env.render()

                action = agent.act(np.expand_dims(obs, 0))[0]
                newobs, reward, done, info = env.step(action)
                total_reward += reward

                if done:
                    break
                else:
                    obs = newobs

        print("Eval after episode #{0}, average reward: {1}".format(
            i, total_reward / NUM_EVALUATIONS))
