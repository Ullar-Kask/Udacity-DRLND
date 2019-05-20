import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dqn_agent import Agent
from monitor import train
from unityagents import UnityEnvironment
import numpy as np


env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# number of states 
state_size = len(env_info.vector_observations[0])
print('Number of states:', state_size)

agent = Agent(state_size, action_size)

scores = train(env, agent)

env.close()


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
fig.savefig('scores.png')
