from ddpg_agent import Agent

from collections import namedtuple, deque
import numpy as np
import torch
import random


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

class MADDPG:
    def __init__(self, actor_state_size, actor_action_size, critic_state_size, critic_action_size):
        super().__init__()
        
        self.agents = [Agent('player1', actor_state_size, actor_action_size, critic_state_size, critic_action_size, device), 
                       Agent('player2', actor_state_size, actor_action_size, critic_state_size, critic_action_size, device)]
        
        self.actor_state_size = actor_state_size
        self.actor_action_size = actor_action_size
        self.critic_state_size = critic_state_size
        self.critic_action_size = critic_action_size
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)
    
    def reset(self):
        """reset all the agents in the MADDPG object"""
        for agent in self.agents:
            agent.reset()
    
    def act(self, obs_all_agents, noise=0.0, clip=True):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise, clip) for agent, obs in zip(self.agents, obs_all_agents)]
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory."""
        self.memory.add(states, actions, rewards, next_states, dones)
    
    def learn(self):
        """update the critics and actors of all the agents """
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            states, states_full, actions, actions_full, next_states, next_states_full, rewards, dones = experiences
            actions_pred, actions_next = [], []
            for idx, agent in enumerate (self.agents):
                actions_pred.append (agent.actor_local (states[idx]))
                actions_next.append (agent.actor_target (next_states[idx]))
            actions_pred_full = torch.cat(actions_pred, dim=1).to(device)
            actions_next_full = torch.cat(actions_next, dim=1).to(device)
            for idx, agent in enumerate (self.agents):
                agent.learn( \
                    states[idx], states_full, \
                    actions[idx], actions_full, \
                    next_states[idx], next_states_full, \
                    actions_pred_full, \
                    actions_next_full, \
                    rewards[idx], \
                    dones[idx])
    
    def save_weights(self, episode=None):
        for agent in self.agents:
            agent.save_weights(episode)
    
    def load_weights(self, episode=None):
        for agent in self.agents:
            agent.load_weights(episode)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state1", "state2", "state_full", "action1", "action2", "action_full", "next_state1", "next_state2", "next_state_full", "reward1", "reward2", "done1", "done2"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience( \
            state[0], \
            state[1], \
            np.reshape(state, -1), \
            action[0], \
            action[1], \
            np.reshape(action, -1), \
            next_state[0], \
            next_state[1], \
            np.reshape(next_state, -1), \
            reward[0], \
            reward[1], \
            done[0], \
            done[1])
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        state1 = torch.from_numpy(np.vstack([e.state1 for e in experiences if e is not None])).float().to(self.device)
        state2 = torch.from_numpy(np.vstack([e.state2 for e in experiences if e is not None])).float().to(self.device)
        states_full = torch.from_numpy(np.vstack([e.state_full for e in experiences if e is not None])).float().to(self.device)
        action1 = torch.from_numpy(np.vstack([e.action1 for e in experiences if e is not None])).float().to(self.device)
        action2 = torch.from_numpy(np.vstack([e.action2 for e in experiences if e is not None])).float().to(self.device)
        actions_full = torch.from_numpy(np.vstack([e.action_full for e in experiences if e is not None])).float().to(self.device)
        next_state1 = torch.from_numpy(np.vstack([e.next_state1 for e in experiences if e is not None])).float().to(self.device)
        next_state2 = torch.from_numpy(np.vstack([e.next_state2 for e in experiences if e is not None])).float().to(self.device)
        next_states_full = torch.from_numpy(np.vstack([e.next_state_full for e in experiences if e is not None])).float().to(self.device)
        reward1 = torch.from_numpy(np.vstack([e.reward1 for e in experiences if e is not None])).float().to(self.device)
        reward2 = torch.from_numpy(np.vstack([e.reward2 for e in experiences if e is not None])).float().to(self.device)
        done1 = torch.from_numpy(np.vstack([e.done1 for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        done2 = torch.from_numpy(np.vstack([e.done2 for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return ([state1, state2], states_full, [action1, action2], actions_full, [next_state1, next_state2], next_states_full, [reward1, reward2], [done1, done2])
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
