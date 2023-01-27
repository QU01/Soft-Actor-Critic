import tensorflow as tf
from tensorflow.keras import * 
import numpy as np

class SoftCritic(tf.keras.Model):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(SoftCritic, self).__init__()
        
        self.linear1 = tf.keras.layers.Dense(hidden_size, input_shape=(None, num_inputs + num_actions), activation='relu')
        self.linear2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.linear3 = tf.keras.layers.Dense(1)
        
        
    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = tf.keras.layers.ReLU()(self.linear1(x))
        x = tf.keras.layers.ReLU()(self.linear2(x))
        x = self.linear3(x)
        return x