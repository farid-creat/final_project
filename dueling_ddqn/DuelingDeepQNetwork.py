import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .ReplayBuffer import ReplayMemory


class DuelingDeepQNetwork(nn.Module):

    def __init__(self, lr, n_actions, input_dims, dropout_p=0.2):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 512)
#        self.fc2 = nn.Linear(512, 512)
        #self.fc3 = nn.Linear(256, 128)


        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f'running on {self.device}')

    def forward(self, state):
        x = F.relu(self.fc1(state))

        # First Residual Block
        #x = F.relu(self.fc2(x))  ########relu
        #x = self.dropout(x)  ##########
        # Second Residual Block
        #x = F.leaky_relu(self.fc3(x), negative_slope=0.1)  ######
        #x = self.dropout(x)  ######

        V = self.V(x)
        A = self.A(x)

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
        # if len(state.shape) == 1:
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

    def learn(self, memory_random, memory_optimal):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        state_r, action_r, reward_r, new_state_r, done_r = memory_random.sample_buffer(self.batch_size // 3)
        state_o, action_o, reward_o, new_state_o, done_o = memory_optimal.sample_buffer(self.batch_size // 3)
        remain = int(self.batch_size - (self.batch_size // 3) * 2)
        state, action, reward, new_state, done = self.memory.sample_buffer(remain)


        state = np.concatenate((state, state_r, state_o), axis=0)
        action = np.concatenate((action, action_r, action_o), axis=0)
        reward = np.concatenate((reward, reward_r, reward_o), axis=0)
        new_state = np.concatenate((new_state, new_state_r, new_state_o), axis=0)
        done = np.concatenate((done, done_r, done_o), axis=0)

        states = T.tensor(state, dtype=T.float).to(self.q_eval.device)
        actions = T.tensor(action, dtype=T.long).to(self.q_eval.device)
        rewards = T.tensor(reward, dtype=T.float).to(self.q_eval.device)
        states_ = T.tensor(new_state, dtype=T.float).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device).bool()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        q_pred = (V_s + (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        V_s_, A_s_ = self.q_next.forward(states_)
        q_next = V_s_ + (A_s_ - A_s_.mean(dim=1, keepdim=True))
        q_next = q_next.detach()
        V_s_eval, A_s_eval = self.q_eval.forward(states_)
        q_eval = V_s_eval + (A_s_eval - A_s_eval.mean(dim=1, keepdim=True))
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()

        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=1.0)

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
        return loss.item()
