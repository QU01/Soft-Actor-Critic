import tensorflow as tf
from tensorflow.keras import * 
import numpy as np
import tensorflow_probability as tfp


class SAC:
    
    def __init__(self, env, value_net, value_target, dqn, dqn2, policy_net, replay_buffer, gamma, learning_rate):
        
        self.env = env
        self.dqn = dqn
        self.dqn2 = dqn2
        self.value_net = value_net
        self.value_target = value_target
        self.policy = policy_net
        self.experience_replay = replay_buffer
        self.counter = 0
        self.update_rate = 80
        self.gamma = gamma
        self.value_net_opt = optimizers.Adam(learning_rate=learning_rate)
        self.critic_1_opt = optimizers.Adam(learning_rate=learning_rate)
        self.critic_2_opt = optimizers.Adam(learning_rate=learning_rate)
        self.policy_opt = optimizers.Adam(learning_rate=learning_rate)


        

    def train(self):

      if len(self.experience_replay.buffer) < self.experience_replay.batch_size:

        return

      states, next_states, actions, rewards, dones = self.experience_replay.sample()

      #Training Critic
      with tf.GradientTape() as tape:
        with tf.GradientTape() as tape2:
          q_values1 = self.dqn(states, actions)
          q_values2 = self.dqn2(states, actions)
          target_value = self.value_target(states)

          target_q_value = rewards.reshape(-1,1) + (1-dones.reshape(-1,1))*self.gamma*target_value

          loss1 = tf.reduce_mean(tf.square(q_values1 - target_q_value))
          loss2 = tf.reduce_mean(tf.square(q_values2 - target_q_value))

      # Obtener los gradientes
      grads1 = tape.gradient(loss1, self.dqn.trainable_variables)
      grads2 = tape2.gradient(loss2, self.dqn2.trainable_variables)

      # Actualizar los parámetros
      self.critic_1_opt.apply_gradients(zip(grads1, self.dqn.trainable_variables))
      self.critic_2_opt.apply_gradients(zip(grads2, self.dqn2.trainable_variables))

      #Training Value Net

      new_actions, log_probs, z, mean, log_std = self.policy.evaluate(states)

      with tf.GradientTape() as tape:

        prior_values = self.value_net(states)
        q_values1 = self.dqn(states, new_actions)
        q_values2 = self.dqn2(states, new_actions)
        if tf.math.reduce_mean(q_values1) >= tf.math.reduce_mean(q_values2):
          q_value = q_values1 
        else:
          q_value = q_values2

        target_values = q_value - log_probs

        value_loss = tf.math.reduce_mean(tf.square(prior_values-target_values))
      
      grads_value = tape.gradient(value_loss, self.value_net.trainable_variables)

      self.value_net_opt.apply_gradients(zip(grads_value,self.value_net.trainable_variables))

      #Training Policy

      with tf.GradientTape() as tape:

        new_actions, log_probs, z, mean, log_std = self.policy.evaluate(states)

        q_values1 = self.dqn(states, new_actions)
        q_values2 = self.dqn2(states, new_actions)
        if tf.math.reduce_mean(q_values1) >= tf.math.reduce_mean(q_values2):
          q_value = q_values1 
        else:
          q_value = q_values2
        policy_loss = tf.reduce_mean(log_probs - q_value)

      grads_policy = tape.gradient(policy_loss, self.policy.trainable_variables)
      self.policy_opt.apply_gradients(zip(grads_policy, self.policy.trainable_variables))

      self.value_target.set_weights(self.value_net.get_weights())