import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .ReplayBuffer import ReplayMemory


class DuelingDeepQNetwork(nn.Module):

    def __init__(self, lr, n_actions, input_dims):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f'running on {self.device}')

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        flat3 = F.relu(self.fc3(flat2))
        V = self.V(flat3)
        A = self.A(flat3)
        return V, A

    def save_model(self, path, name):
        os.makedirs(path, exist_ok=True)
        T.save(self.state_dict(), os.path.join(path, name))

    def load_model(self, path, device, name):
        self.load_state_dict(T.load(os.path.join(path, name), map_location=device))


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace

        self.learn_step_counter = 0

        self.action_space = [i for i in range(self.n_actions)]

        self.memory = ReplayMemory(mem_size, input_dims)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims)

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims)
    def reset_memory(self):
        self.memory.reset();

    def predict_q_value(self, state, action):
        state = T.tensor([state], dtype=T.float).to(self.q_eval.device)
        #if len(state.shape) == 1:
        #    state = state.unsqueeze(0)  # Add a batch dimension
        V, A = self.q_eval.forward(state)
        q_value = V + (A - A.mean(dim=1, keepdim=True))
        return q_value[0, action].item()  # Return the Q-value for the specific action

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transitions(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self, path):
        self.q_eval.save_model(path=path, name="q_eval")
        self.q_next.save_model(path=path, name="q_next")

    def load_models(self, path):
        self.q_eval.load_model(path=path, device=self.q_eval.device, name="q_eval")
        self.q_next.load_model(path=path, device=self.q_next.device, name="q_next")

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device).bool()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
        return loss.item()
