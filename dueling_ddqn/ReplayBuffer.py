import random
import numpy as np

import torch


class ReplayMemory:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transitions(self, state, action, reward ,next_state , done):
        index = self.mem_cntr% self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr +=1

    def is_full(self):
        return self.mem_size == min(self.mem_cntr , self.mem_size)

    def reset(self):
        self.mem_cntr = 0


    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr , self.mem_size)
        batch = np.random.choice(max_mem , batch_size , replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards ,states_, terminal
