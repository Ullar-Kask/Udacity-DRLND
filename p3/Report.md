# Project Report

### Overview

The implementation is based on the MADDPG (multi-agent DDPG) network architecture consisting of two DDPG agents (player1 and player2) with actor and critic networks having both two hidden layers.
Batch normalization is used before and after the first hidden layer in actor and critic networks,
and gradient clipping is used when training the critic networks.

### Folder Structure

The project folder contains the following files and folders:

- `src/` - Project source code folder
- `README.md` - Project readme file
- `player1_actor.pt` - Saved weights of the trained player1 actor network
- `player2_actor.pt` - Saved weights of the trained player2 actor network
- `player1_critic.pt` - Saved weights of the trained player1 critic network
- `player2_critic.pt` - Saved weights of the trained player2 critic network
- `scores.png` - Scores plot
- `Report.md` - Project report

The source code in `src/` folder is laid out as follows:
- `main.py` - main module that starts the environment, creates the agent and starts the training process
- `model.py` - defines the network model that is used by DDPG agents
- `ddpg_agent.py` - defines a DDPG agent
- `maddpg.py` - manages the swarm of DDPG agents
- `monitor.py` - defines how the swarm and the environment interact with each other

### Learning Algorithm

This is an implementation of continuous action actor-critic algorithm that uses
Multi-Agent Deterministic Policy Gradient (MADDPG) method
for updating the actor and critic networks for each agent, 
uses stochastic behaviour policy for exploration (Ornstein-Uhlenbeck) and 
outputs a deterministic target policy for each agent.

Each agent's actor network is policy estimator that consists of a neural network 
taking into input the state of the particular agent and outputs an action. 
The agent's critic network is policy evaluator that consists of a neural network
taking into input the states of all agents and corresponding actions from all agents and 
outputs the state-action value function for the particular agent.
The output of the agent's critic drives learning in both the actor and the critic of the agent.

The experience replay buffer is used during agents training, random minibatches of experiences are sampled from the buffer
and used for learning in order to break up the temporal correlations within different training episodes.

Other techniques used to overcome instablilies during training are gradient clipping, soft update of target model parameters and batch normalization.

**Experience Replay**

When an agent interacts with the environment, the sequence of experience tuples (state, action, reward, next state, flag to indicate if episode is completed) is stored in the 
replay buffer (the buffer size is controlled by a hyperparameter). Experience replay tuples sampled from the buffer at random are used to prevent action values from oscillating or diverging catastrophically.
The tuples are gradually added to the buffer as agents interact with the environment.

Replaying experience samples from the replay buffer help to break harmful correlations, learn from
individual tuples multiple times, recall rare occurrences, and in general make better use of gathered experience.


### Hyperparameters

- Both DDPG agents have the same DDPG network architecture
- Each DDPG actor and critic network consists of two fully connected hidden layers with 400 and 300 neurons respectively
- All layers use `ReLu` activation functions, except for the output layers of the actor networks that use `tanh` activation function
- Replay buffer size: `BUFFER_SIZE = int(1e5)`
- Minibatch size: `BATCH_SIZE = 128`
- Discount factor: `GAMMA = 0.99`
- Soft update factor of target parameters: `TAU = 1e-3`
- Learning rate of the actor network: `LR_ACTOR = 1e-3`
- Learning rate of the critic network: `LR_CRITIC = 1e-3`
- Weight decay of the critic network: `WEIGHT_DECAY = 0`
- Each episode consists of `max_t=1000` timesteps
- Networks are updated after every `LEARN_EVERY=1` timesteps, `LEARN_BATCH=5` times in a go

### Running

To start training enter on the command line:
```sh
$ python main.py
```

The training is completed after 615 episodes with the final average score of +0.52.

After the training is completed the files `player1_actor.pt`, `player2_actor.pt`, `player1_critic.pt` and `player2_critic.pt` are created automatically in the subfolder `trained_weights/`.
The files contain the weights of the trained actor and critic networks.

Also a scores plot is created in the file `scores.png`.

### Training process

    Episode 10/5000 || Average score 0.00
    Episode 20/5000 || Average score 0.01
    Episode 30/5000 || Average score 0.01
    Episode 40/5000 || Average score 0.01
    Episode 50/5000 || Average score 0.01
    Episode 60/5000 || Average score 0.01
    Episode 70/5000 || Average score 0.01
    Episode 80/5000 || Average score 0.01
    Episode 90/5000 || Average score 0.01
    Episode 100/5000 || Average score 0.01
    Episode 110/5000 || Average score 0.01
    Episode 120/5000 || Average score 0.02
    Episode 130/5000 || Average score 0.02
    Episode 140/5000 || Average score 0.02
    Episode 150/5000 || Average score 0.03
    Episode 160/5000 || Average score 0.03
    Episode 170/5000 || Average score 0.03
    Episode 180/5000 || Average score 0.03
    Episode 190/5000 || Average score 0.04
    Episode 200/5000 || Average score 0.04
    Episode 210/5000 || Average score 0.04
    Episode 220/5000 || Average score 0.05
    Episode 230/5000 || Average score 0.05
    Episode 240/5000 || Average score 0.05
    Episode 250/5000 || Average score 0.06
    Episode 260/5000 || Average score 0.06
    Episode 270/5000 || Average score 0.07
    Episode 280/5000 || Average score 0.07
    Episode 290/5000 || Average score 0.07
    Episode 300/5000 || Average score 0.07
    Episode 310/5000 || Average score 0.07
    Episode 320/5000 || Average score 0.07
    Episode 330/5000 || Average score 0.07
    Episode 340/5000 || Average score 0.07
    Episode 350/5000 || Average score 0.07
    Episode 360/5000 || Average score 0.07
    Episode 370/5000 || Average score 0.08
    Episode 380/5000 || Average score 0.08
    Episode 390/5000 || Average score 0.09
    Episode 400/5000 || Average score 0.09
    Episode 410/5000 || Average score 0.11
    Episode 420/5000 || Average score 0.12
    Episode 430/5000 || Average score 0.12
    Episode 440/5000 || Average score 0.13
    Episode 450/5000 || Average score 0.15
    Episode 460/5000 || Average score 0.17
    Episode 470/5000 || Average score 0.18
    Episode 480/5000 || Average score 0.19
    Episode 490/5000 || Average score 0.19
    Episode 500/5000 || Average score 0.20
    Episode 510/5000 || Average score 0.20
    Episode 520/5000 || Average score 0.21
    Episode 530/5000 || Average score 0.21
    Episode 540/5000 || Average score 0.20
    Episode 550/5000 || Average score 0.19
    Episode 560/5000 || Average score 0.19
    Episode 570/5000 || Average score 0.21
    Episode 580/5000 || Average score 0.23
    Episode 590/5000 || Average score 0.34
    Episode 600/5000 || Average score 0.38
    Episode 610/5000 || Average score 0.45
    Environment solved in 615 episodes. Average score: 0.52


### Ideas for Future Work

The neural network can be improved by implementing the following:
- Prioritized Experience Replay ([PER](https://arxiv.org/pdf/1803.00933.pdf))
- Instead of DDPG try a different network algorithm, e.g. [TD3](https://arxiv.org/pdf/1802.09477.pdf) in multi-agent mode.