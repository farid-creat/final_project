import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from .Actor import Actor
from .Critic import Critic
from .Memory import Memory
import numpy as np
import os
class DDPGagent:
    def __init__(self , num_states , num_actions , hidden_size = 512 , actor_learning = 1e-2 , critic_learning = 1e-2
                 , gamma = 0.95 , tau = 1e-2 , max_memmory_size = 100000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(f'running on {device}')
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.actor = Actor(self.num_states , hidden_size , self.num_actions).to(device)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).to(device)

        self.critic = Critic(self.num_states + num_actions, hidden_size, 1).to(device)
        self.critic_target = Critic(self.num_states + num_actions, hidden_size, 1).to(device)

        for target_params , params in zip(self.actor_target.parameters() , self.actor.parameters()):
            target_params.data.copy_(params.data)
        for target_params , params in zip(self.critic_target.parameters() , self.critic.parameters()):
            target_params.data.copy_(params.data)

        self.memory = Memory(max_memmory_size)

        self.critic_criterion = nn.MSELoss()

        self.actor_optimizer = optim.Adam(self.actor.parameters() , lr=actor_learning)
        self.critic_optimizer = optim.Adam(self.critic.parameters() , lr= critic_learning)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).to(self.device).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0,:]
        return action

    def update(self , batch_size , out_memory = None , percent_outMemory = 0):
        amount_from_out = batch_size*percent_outMemory//100
        states , actions , rewards , next_states , dones = self.memory.sample(batch_size - amount_from_out)
        if out_memory is not None:
            states_out, actions_out, rewards_out, next_states_out, dones_out = out_memory.sample(amount_from_out)
            states = np.concatenate((states, states_out), axis=0)
            actions = np.concatenate((actions, actions_out), axis=0)
            rewards = np.concatenate((rewards, rewards_out), axis=0)
            next_states = np.concatenate((next_states, next_states_out), axis=0)
            dones = np.concatenate((dones, dones_out), axis=0)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)


        Qvals = self.critic.forward(states,actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states,next_actions.detach())
#        rewards = rewards.view(next_Q.size())
#        dones = dones.view(next_Q.size())

        Qprime = []
        for i in range(batch_size):
            Qprime.append(rewards[i] + self.gamma*next_Q[i] * (1-dones[i]))
        Qprime = torch.tensor(Qprime).to(self.device)
        Qprime = Qprime.view(Qvals.size())
        #Qprime = rewards + self.gamma * next_Q * (1 - dones)
        critic_loss = self.critic_criterion(Qvals , Qprime)
        policy_objective = -self.critic.forward(states , self.actor.forward(states)).mean()

        #print(f'Critic Loss: {critic_loss}, Policy Loss: {policy_loss}')
        self.actor_optimizer.zero_grad()
        policy_objective.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        for target_params , params in zip(self.actor_target.parameters() , self.actor.parameters()):
            target_params.data.copy_(params.data*self.tau + target_params.data * (1-self.tau))
        for target_params , params in zip(self.critic_target.parameters() , self.critic.parameters()):
            target_params.data.copy_(params.data * self.tau + target_params.data * (1 - self.tau))
        return critic_loss.detach().cpu() , policy_objective.detach().cpu()


    def save_models(self, path):
        # Create directories if they don't exist
        os.makedirs(path, exist_ok=True)

        # Save actor and critic models
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(path, 'actor_target.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        torch.save(self.critic_target.state_dict(), os.path.join(path, 'critic_target.pth'))

    def load_models(self, path):
        # Load actor and critic models
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth'), map_location=self.device))
        self.actor_target.load_state_dict(torch.load(os.path.join(path, 'actor_target.pth'), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth'), map_location=self.device))
        self.critic_target.load_state_dict(torch.load(os.path.join(path, 'critic_target.pth'), map_location=self.device))

