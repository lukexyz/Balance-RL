# Custom reward functions for OpenAI reinforcement learning

import numpy as np
from render_utils import *
import math


def discount_rewards(rewards, discount_rate, obs):
    """ Modify reward to improve historic behaviour """

    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    phi = 1.0

    # print('step, obs[step], rewards[step], drift_penalty, cumulative_rewards, discounted_rewards[step]')

    for step in reversed(range(len(rewards))):
        # Prevent DRIFT by adding penalty (distance from centre)
        drift_penalty = phi * abs(obs[step][0])

        # drift_penalty = max(0.2, phi * abs(obs[step][0]))
        # drift_penalty = 0.5 if abs(obs[step][0]) > 0.3 else 0

        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate - drift_penalty
        discounted_rewards[step] = cumulative_rewards
        # print(step, obs[step], rewards[step], drift_penalty, cumulative_rewards, discounted_rewards[step])

    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, all_obs, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate, all_obs[i])
                              for i, rewards in enumerate(all_rewards)]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


"""
New reward function:
    Add penalty for (-theta) and (+momentum)
    or the inverse: (+theta) and (-momentum)
"""