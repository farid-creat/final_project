import random
from collections import deque
import pickle
import numpy as np
class Memory:
    def __init__(self , max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.size = 0
    def push(self , state , action , reward , next_state , done):
        experience = (state , action , reward , next_state , done)
        self.buffer.append(experience)
        if self.size<self.max_size:
            self.size+=1
    def sample (self, batch_size):
        actual_size = min(batch_size ,self.size)
        batch = random.sample(self.buffer, actual_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    def len(self):
        return self.size

    def save(self, file_path):
        """Save the buffer and size to a file."""
        with open(file_path, 'wb') as f:
            # Save the buffer and size to the file
            pickle.dump((self.buffer, self.size), f)

    def load(self, file_path):
        """Load the buffer and size from a file."""
        with open(file_path, 'rb') as f:
            # Load the buffer and size from the file
            self.buffer, self.size = pickle.load(f)
