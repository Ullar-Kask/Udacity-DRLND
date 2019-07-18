import numpy as np
import random
import copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor network
LR_CRITIC = 1e-3        # learning rate of the critic network
WEIGHT_DECAY = 0.0      # L2 weight decay

weights_file_folder = 'trained_weights'
actor_weights_file = 'actor.pt'
critic_weights_file = 'critic.pt'

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, id, actor_state_size, actor_action_size, critic_state_size, critic_action_size, device):
        """Initialize an Agent object.
        
        Params
        ======
            id (string): agent's id
            actor_state_size (int): dimension of actor's each state
            actor_action_size (int): dimension of actor's each action
            critic_state_size (int): dimension of critic's each state
            critic_action_size (int): dimension of critic's each action
        """
        self.id = id
        self.actor_state_size = actor_state_size
        self.actor_action_size = actor_action_size
        self.critic_state_size = critic_state_size
        self.critic_action_size = critic_action_size
        self.device = device
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(actor_state_size, actor_action_size).to(self.device)
        self.actor_target = Actor(actor_state_size, actor_action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(critic_state_size, critic_action_size).to(self.device)
        self.critic_target = Critic(critic_state_size, critic_action_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Noise process
        self.noise = OUNoise(actor_action_size)
    
    def act(self, state, noise=0.0, clip=True):
        """Returns actions for given state as per current policy."""
        state = torch.unsqueeze(torch.from_numpy(state).float(), 0).detach().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = torch.squeeze(self.actor_local(state).cpu()).data.numpy()
        self.actor_local.train()
        if noise != 0.0:
            action += noise*self.noise.sample()
            #action += noise*np.random.randn(self.actor_action_size)
        if clip:
            return np.clip(action, -1, 1)
        return action
    
    def reset(self):
        self.noise.reset()
    
    def learn(self, state, state_full, action, action_full, next_state, next_state_full, action_pred_full, action_next_full, reward, done):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_state_full, action_next_full)
        # Compute Q targets for current states (y_i)
        Q_targets = reward + (GAMMA * Q_targets_next * (1 - done))
        # Compute critic loss
        Q_expected = self.critic_local(state_full, action_full)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actor_loss = -self.critic_local(state_full, action_pred_full).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
        return actor_loss, critic_loss
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
    
    def save_weights(self, episode=None):
        if episode == None:
            prefix = self.id
        else:
            prefix = self.id + '-' + str(episode)
        torch.save(self.actor_local.state_dict(), weights_file_folder + '/' + prefix + '-' + actor_weights_file)
        torch.save(self.critic_local.state_dict(), weights_file_folder + '/' + prefix + '-' + critic_weights_file)
    
    def load_weights(self, episode=None):
        if episode == None:
            prefix = self.id
        else:
            prefix = self.id + '-' + str(episode)
        self.actor_local.load_state_dict(torch.load(weights_file_folder + '/' + prefix + '-' + actor_weights_file))
        self.critic_local.load_state_dict(torch.load(weights_file_folder + '/' + prefix + '-' + critic_weights_file))

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, size, mu=0., theta=0.15, sigma=0.5):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
    
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
