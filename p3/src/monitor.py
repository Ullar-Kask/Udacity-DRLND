from maddpg import MADDPG

from collections import deque
import numpy as np

# Number of timesteps between invoking learning on agents
LEARN_EVERY = 1
# How many times to learn in a go
LEARN_BATCH = 5

# Save weights after every so many episodes
SAVE_EVERY = 500

def train(env, num_episodes=5000, max_t=1000, warmup_episodes=0):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of the environment
    - num_episodes: maximum number of episodes of agent-environment interaction
    - max_t: maximum number of timesteps per episode
    - warmup_episodes: how many episodes to explore and collect samples before learning begins
    
    Returns
    =======
    - scores: list containing received rewards
    """
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    
    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 1.0
    noise_reduction = 0.9999
    
    # list containing max scores from each episode
    episode_scores = []
    # last 100 scores
    scores_window = deque(maxlen=100)
    mean_score = 0.0
    
    maddpg = MADDPG(state_size, action_size, num_agents*state_size, num_agents*action_size)
    
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # reset the environment and begin the episode
        env_info = env.reset(train_mode=True)[brain_name]
        maddpg.reset()
        
        # get the current state (for each agent)
        states = env_info.vector_observations
        
        # initialize the score (for each agent)
        scores = np.zeros(num_agents)
        
        for t in range(max_t):
            # select an action (for each agent)
            if i_episode > warmup_episodes:
                actions = maddpg.act(states, noise)
                noise *= noise_reduction
            else:
                # Collect random samples to explore and fill the replay buffer
                actions = np.random.uniform(-1,1,(num_agents,action_size))
            
            # send all actions to the environment
            env_info = env.step(actions)[brain_name]
            
            # get next state (for each agent)
            next_states = env_info.vector_observations
            
            # get reward (for each agent)
            rewards = env_info.rewards
            
            # see if episode finished
            dones = env_info.local_done
            
            # agents perform internal updates based on sampled experience
            maddpg.step(states, actions, rewards, next_states, dones)
            
            # roll over states to next time step
            states = next_states
            
            # learn when time is right
            if t % LEARN_EVERY == 0 and i_episode > warmup_episodes:
                for _ in range(LEARN_BATCH):
                    maddpg.learn()
            
            # update the score (for each agent)
            scores += rewards
            
            # exit loop if episode finished
            if np.any(dones):
                break
        
        episode_max_score = np.max(scores)
        episode_scores.append(episode_max_score)
        
        if i_episode > warmup_episodes:
            # save final score
            scores_window.append(episode_max_score)
            mean_score = np.mean(scores_window)
            # monitor progress
            if i_episode % 10 == 0:
                print("\rEpisode {:d}/{:d} || Average score {:.2f}".format(i_episode, num_episodes, mean_score))
        else:
            print("\rWarmup episode {:d}/{:d}".format(i_episode, warmup_episodes), end="")
        
        if i_episode % SAVE_EVERY == 0 and i_episode > warmup_episodes:
            maddpg.save_weights(i_episode)
        
        # check if task is solved
        if i_episode >= 100 and mean_score >= 0.5:
            print('\nEnvironment solved in {:d} episodes. Average score: {:.2f}'.format(i_episode, mean_score))
            maddpg.save_weights()
            break
    if i_episode == num_episodes: 
        print("\nGame over. Too bad! Final score {:.2f}\n".format(mean_score))
    return episode_scores
