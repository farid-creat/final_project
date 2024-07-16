import numpy as np


class OUNoise():
    def __init__(self , action_dim , action_low, action_high , mu=0.0 , theta = 0.15 , max_sigma = 0.3 , min_sigma = 0.3 , decay_period = 100_000):
        self.state = None
        self.mu = mu
        self.theta = theta
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.sigma = self.max_sigma
        self.reset()
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma*np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self , action , t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1,t/self.decay_period)
        return np.clip(action + ou_state , self.action_low , self.action_high)


