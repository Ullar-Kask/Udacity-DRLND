# Project Report

### Overview

The implementation is based on the DDPG network architecture with actor and critic networks having both two hidden layers.
Batch normalization is used before calling the activation function after the first hidden layer,
and gradient clipping is used when training the critic network. The task can solved without batch normalization, but due to instabilities it would take about 30% more episodes.

### Folder Structure

The project folder contains the following files and folders:

- `src/` - Project source code folder
- `README.md` - Project readme file
- `weights_actor.pt` - Saved weights of the trained actor network
- `weights_critic.pt` - Saved weights of the trained critic network
- `scores.png` - Scores plot
- `Report.md` - Project report

The source code in `src/` folder is laid out as follows:
- `main.py` - main module that starts the environment, creates the agent and starts the training process
- `model.py` - defines the network model
- `ddpg_agent.py` - defines the agent
- `monitor.py` - defines how the agent and the environment interact with each other

### Learning Algorithm

This is an implementation of continuous action actor-critic algorithm that uses Deterministic Policy Gradient (DDPG) method
for updating the actor and critic networks, uses stochastic behaviour policy for exploration (Ornstein-Uhlenbeck) and outputs a deterministic target policy.

The actor network is policy estimator that consists of a neural network taking into input the state and outputs the action. 
THe critic network is policy evaluator that consists of a neural network taking into input the state and corresponding action and outputs the state-action value function.
The output of the critic drives learning in both the actor and the critic.

The experience replay buffer is used during agent training, random minibatches of experiences are sampled from the buffer
and used for learning in order to break up the temporal correlations within different training episodes.

Other techniques used to overcome instablilies during training are gradient clipping, soft update of target model parameters and batch normalization.


**Experience Replay**

When the agent interacts with the environment, the sequence of experience tuples (state, action, reward, next state, flag to indicate if episode is completed) is stored in the 
replay buffer (the buffer size is controlled by a hyperparameter). Experience replay tuples sampled from the buffer at random are used to prevent action values from oscillating or diverging catastrophically.
The tuples are gradually added to the buffer as the agent is interacting with the environment.

Replaying experience samples from the replay buffer helps to break harmful correlations, learn from
individual tuples multiple times, recall rare occurrences, and in general make better use of gathered experience.


### Hyperparameters

- Both actor and critic networks consist of 2 fully connected hidden layers with 400 and 300 neurons respectively
- All layers use `ReLu` activation functions, except for the output layer of the actor network that uses `tanh` activation function
- Replay buffer size: `BUFFER_SIZE = int(1e6)`
- Minibatch size: `BATCH_SIZE = 1024`
- Discount factor: `GAMMA = 0.99`
- Soft update factor of target parameters: `TAU = 1e-3`
- Learning rate of the actor network: `LR_ACTOR = 9.99e-4`
- Learning rate of the critic network: `LR_CRITIC = 9.99e-4`
- Weight decay of the critic network: `WEIGHT_DECAY = 0`
- Each episode consists of `max_t=1000` timesteps
- Networks are updated `learn_batch=10` times after every `learn_step=20` timesteps

Some observations:
1. Batch size should be relatively large for faster convergence, as is `1024` in our case.
2. Equal learning rates (of actor and critic networks) seem to work better.
3. The training process is susceptible to changes in the order of `1e-6` in the learning rates, so it is easy to overshoot (or undershoot).


### Running

To start training enter on the command line:
```sh
$ python main.py
```

The training should be completed after 100 episodes with the final average score of +33.12.
(The average score of +30.13 is achieved after episode 64, but strictly speaking it's the average score of 64 episodes, so the training continues until the episode 100).

After the training is completed files `weights_actor.pt` and `weights_critic.pt` are created automatically that contains the weights of the trained actor and critic networks respectively.
Also a scores plot is created in the file `scores.png`.

### Training process

Episode 1/500 || Average score 0.61
Episode 2/500 || Average score 0.59
Episode 3/500 || Average score 0.60
Episode 4/500 || Average score 0.64
Episode 5/500 || Average score 0.78
Episode 6/500 || Average score 0.89
Episode 7/500 || Average score 1.07
Episode 8/500 || Average score 1.22
Episode 9/500 || Average score 1.33
Episode 10/500 || Average score 1.57
Episode 11/500 || Average score 1.80
Episode 12/500 || Average score 2.28
Episode 13/500 || Average score 3.00
Episode 14/500 || Average score 3.66
Episode 15/500 || Average score 4.65
Episode 16/500 || Average score 5.95
Episode 17/500 || Average score 7.24
Episode 18/500 || Average score 8.61
Episode 19/500 || Average score 10.10
Episode 20/500 || Average score 11.38
Episode 21/500 || Average score 12.63
Episode 22/500 || Average score 13.77
Episode 23/500 || Average score 14.85
Episode 24/500 || Average score 15.85
Episode 25/500 || Average score 16.77
Episode 26/500 || Average score 17.63
Episode 27/500 || Average score 18.43
Episode 28/500 || Average score 19.16
Episode 29/500 || Average score 19.83
Episode 30/500 || Average score 20.47
Episode 31/500 || Average score 21.06
Episode 32/500 || Average score 21.62
Episode 33/500 || Average score 22.15
Episode 34/500 || Average score 22.62
Episode 35/500 || Average score 23.08
Episode 36/500 || Average score 23.51
Episode 37/500 || Average score 23.92
Episode 38/500 || Average score 24.31
Episode 39/500 || Average score 24.68
Episode 40/500 || Average score 25.02
Episode 41/500 || Average score 25.35
Episode 42/500 || Average score 25.68
Episode 43/500 || Average score 25.98
Episode 44/500 || Average score 26.26
Episode 45/500 || Average score 26.54
Episode 46/500 || Average score 26.81
Episode 47/500 || Average score 27.06
Episode 48/500 || Average score 27.30
Episode 49/500 || Average score 27.51
Episode 50/500 || Average score 27.73
Episode 51/500 || Average score 27.95
Episode 52/500 || Average score 28.16
Episode 53/500 || Average score 28.35
Episode 54/500 || Average score 28.54
Episode 55/500 || Average score 28.71
Episode 56/500 || Average score 28.89
Episode 57/500 || Average score 29.05
Episode 58/500 || Average score 29.22
Episode 59/500 || Average score 29.38
Episode 60/500 || Average score 29.54
Episode 61/500 || Average score 29.69
Episode 62/500 || Average score 29.84
Episode 63/500 || Average score 29.98
Episode 64/500 || Average score 30.13
Episode 65/500 || Average score 30.27
Episode 66/500 || Average score 30.40
Episode 67/500 || Average score 30.52
Episode 68/500 || Average score 30.64
Episode 69/500 || Average score 30.76
Episode 70/500 || Average score 30.88
Episode 71/500 || Average score 30.99
Episode 72/500 || Average score 31.11
Episode 73/500 || Average score 31.21
Episode 74/500 || Average score 31.32
Episode 75/500 || Average score 31.40
Episode 76/500 || Average score 31.49
Episode 77/500 || Average score 31.58
Episode 78/500 || Average score 31.66
Episode 79/500 || Average score 31.75
Episode 80/500 || Average score 31.84
Episode 81/500 || Average score 31.91
Episode 82/500 || Average score 31.98
Episode 83/500 || Average score 32.04
Episode 84/500 || Average score 32.11
Episode 85/500 || Average score 32.19
Episode 86/500 || Average score 32.26
Episode 87/500 || Average score 32.33
Episode 88/500 || Average score 32.40
Episode 89/500 || Average score 32.47
Episode 90/500 || Average score 32.53
Episode 91/500 || Average score 32.60
Episode 92/500 || Average score 32.66
Episode 93/500 || Average score 32.72
Episode 94/500 || Average score 32.78
Episode 95/500 || Average score 32.83
Episode 96/500 || Average score 32.89
Episode 97/500 || Average score 32.95
Episode 98/500 || Average score 33.00
Episode 99/500 || Average score 33.06
Episode 100/500 || Average score 33.12


### Ideas for Future Work

The neural network can be improved by implementing the following:
- Prioritized Experience Replay
- Instead of DDPG try a different network algorithm, e.g. [Soft Actor Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html) (SAC)