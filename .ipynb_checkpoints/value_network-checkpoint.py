import tensorflow as tf
from tensorflow.keras import * 
import numpy as np


class ValueNetwork(tf.keras.Model):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = layers.Dense(hidden_dim, activation='relu', input_shape=(None, state_dim))
        self.linear2 = layers.Dense(hidden_dim, activation='relu')
        self.linear3 = layers.Dense(1)

        
    def call(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        return x