from collections import deque
import numpy as np

def train(env, agent, num_episodes=1000, max_t=1000, window=100, eps_start=1.0, eps_end=0.001, eps_decay=0.95):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of the environment
    - agent: instance of class Agent
    - num_episodes: number of episodes of agent-environment interaction
    - max_t: maximum number of timesteps per episode
    - window: number of episodes to consider when calculating average rewards
    - eps_start: starting value of epsilon, for epsilon-greedy action selection
    - eps_end: minimum value of epsilon
    - eps_decay: multiplicative factor (per episode) for decreasing epsilon
    
    Returns
    =======
    - scores: list containing received rewards
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # list containing scores from each episode
    scores = []
    # last 100 scores
    scores_window = deque(maxlen=window)
    
    # initialize epsilon
    eps = eps_start
    
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        
        # initialize the score
        score = 0
        
        for t in range(max_t):
            # agent selects an action
            action = agent.act(state, eps)
            
            # agent performs the selected action
            env_info = env.step(action)[brain_name]
            
            # get the next state
            next_state = env_info.vector_observations[0]
            
            # get the reward
            reward = env_info.rewards[0]
            
            # see if episode has finished
            done = env_info.local_done[0]
            
            # agent performs internal updates based on sampled experience
            agent.step(state, action, reward, next_state, done)
            
            # update the score
            score += reward
            
            # update the state (s <- s') to next time step
            state = next_state
            
            if done:
                break
        
        # save final score
        scores_window.append(score)
        scores.append(score)
        
        # decrease epsilon
        eps = max(eps_end, eps_decay*eps)
        
        # monitor progress
        print("\rEpisode {:d}/{:d} || Average score {:.2f} || eps={:.5f}".format(i_episode, num_episodes, np.mean(scores_window), eps), end="")
        
        # check if task is solved
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes. Final score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            agent.save_weights()
            break
    if i_episode == num_episodes: 
        print("\nEnvironment not solved. Final score {:.2f}\n".format(np.mean(scores_window)))
    return scores
