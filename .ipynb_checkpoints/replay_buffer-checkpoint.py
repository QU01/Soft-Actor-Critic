import numpy as np

class ReplayBuffer:
    
    def __init__(self, max_size, batch_size):
        
        self.buffer = []
        self.max_size = max_size
        self.batch_size = batch_size
    
    def append(self, state, next_state, action, reward, done, info):
        
        sample = (state, next_state, action, reward, done, info)
        
        if len(self.buffer) >= self.max_size:
            
            self.buffer.pop(0)
        
        self.buffer.append(sample)
        
    def sample(self):
        
        idxs = np.random.choice(np.arange(len(self.buffer)), size = self.batch_size)
        
        states = np.array([self.buffer[i][0] for i in idxs])
        next_states = np.array([self.buffer[i][1] for i in idxs])
        actions = np.array([self.buffer[i][2] for i in idxs])
        rewards = np.array([self.buffer[i][3] for i in idxs])
        dones = np.array([self.buffer[i][4] for i in idxs])
            
        return states, next_states, actions, rewards, dones