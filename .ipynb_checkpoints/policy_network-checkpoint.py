import tensorflow as tf
from tensorflow.keras import * 
import numpy as np
import tensorflow_probability as tfp

class Actor(tf.keras.Model):
    def __init__(self, state_dim, num_actions, hidden_size, action_high=-1, action_low=1):
        super(Actor, self).__init__()
        
        self.action_high = action_high
        self.action_low = action_low
        
        self.linear1 = layers.Dense(hidden_size, activation='relu', input_shape=(None, state_dim))
        self.linear2 = layers.Dense(hidden_size, activation='relu')
        
        self.mean_linear = layers.Dense(num_actions)
        self.log_std_linear = layers.Dense(num_actions)
        
    def call(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, self.action_low, self.action_high)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.call(state)
        std = tf.exp(log_std)
        
        normal = tfp.distributions.Normal(mean, std)
        z = normal.sample()
        action = tf.tanh(z)
        
        log_prob = normal.log_prob(z) - tf.math.log(1 - tf.pow(action, 2) + epsilon)
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
        
        return action, log_prob, z, mean, log_std
        
    def get_action(self, state):
        mean, log_std = self.call(tf.expand_dims(state, axis=0))
        std = tf.exp(log_std)
        
        normal = tfp.distributions.Normal(mean, std)
        z      = normal.sample()
        action = tf.tanh(z)
        
        return action.numpy()[0]