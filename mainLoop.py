import mujoco_py
import gym

import tensorflow as tf
from tensorflow.keras import * 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_probability as tfp

from policy_network import Actor
from value_network import ValueNetwork
from SoftDQN import SoftCritic
from SAC_agent import SAC
import ReplayBuffer from replay_buffer

env = gym.make("HalfCheetah-v2")

value_net = ValueNetwork(17, 32)
value_target = ValueNetwork(17, 32)
value_target.set_weights(value_net.get_weights())

critic_1 = SoftCritic(17, 6, 32)
critic_2 = SoftCritic(17, 6, 32)

actor = Actor(17, 6, 32)

experience_replay = ReplayBuffer(100000, 64)

gamma = 0.99
episodes = 200 
learning_rate = 3e-4

sac = SAC(env, value_net, value_target, critic_1, critic_2, actor, experience_replay, gamma, learning_rate)

returns = []
avg_returns = []
total_steps = 0

for episode in range(episodes):

  Return = 0

  done = False
  state = env.reset()
  steps = 0

  while not done:

    action = sac.policy.get_action(state)
    next_state, reward, done, info = env.step(action)
    Return += reward
    experience_replay.append(state, next_state, action, reward, done, info)
    state = next_state
    steps += 1

  sac.train()

  returns.append(Return)
  avg_return = np.mean(returns[-10:])
  avg_returns.append(avg_return)

  total_steps += steps

  print("Episode: " + str(episode)+"/"+str(episodes) + " return of "+ str(Return) + " average reward of: " + str(avg_return) +  "in " + str(steps) + " steps")

  if episode % 10 == 0:
    sac.dqn.save_weights(f"critic1_{ episode }", save_format='tf')
    sac.dqn2.save_weights(f"critic2_{ episode }", save_format='tf')
    
    sac.value_net.save_weights(f"value_{ episode }", save_format='tf')

    sac.policy.save_weights(f"actor_{ episode }", save_format='tf')