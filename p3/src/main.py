import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from monitor import train
from unityagents import UnityEnvironment
import numpy as np
import torch


random_seed=11

np.random.seed(random_seed)
torch.manual_seed(random_seed)

env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")

scores = train(env)

env.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
fig.savefig('scores.png')
