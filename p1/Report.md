# Project Report

### Overview

The implementation is based on fixed Q-target deep Q-network (DQN) with 2 fully connected hidden layers.


### Folder Structure

The project folder contains the following files and folders:

- `src/` - Project source code folder
- `README.md` - Project readme file
- `weights.pt` - Saved weights of the trained network
- `scores.png` - Scores plot
- `Report.md` - Project report

The source code in `src/` folder is laid out as follows:
- `main.py` - main module that starts the environment, creates the agent and starts the training process
- `model.py` - defines the network model
- `dqn_agent.py` - defines the agent
- `monitor.py` - defines how the agent and the environment interact with each other

### Learning Algorithm

Reinforcement learning tends to be unstable when neural networks are used to represent the action values.
To overcome these instabilities the following two techniques are used in the project:
- Experience Replay
- Fixed Q-Targets

**Experience Replay**

When the agent interacts with the environment, the sequence of experience tuples (state, action, reward, next state, flag to indicate if episode is completed) is stored in the 
replay buffer (the buffer size is controlled by a hyperparameter). Experience replay tuples sampled from the buffer at random are used to prevent action values from oscillating or diverging catastrophically.
The tuples are gradually added to the buffer as the agent is interacting with the environment.

Replaying experience samples from the replay buffer helps to break harmful correlations, learn from
individual tuples multiple times, recall rare occurrences, and in general make better use of gathered experience.

**Fixed Q-Targets**

In the fixed Q-targets framework the deep Q-learning algorithm uses two separate neural networks with identical architectures (primary and target networks).
The weights of the primary network are updated at every timestep, but weights of the target network are updated less often (after every n timesteps controlled by a hyperparameter).
The temporal difference targets (TD targets) of an action are calculated based on the values stored in the target network.
As these weights are updated less frequently we are able to avoid a form of harmful correlation whereby we shift the parameters of the network based
on a constantly moving target.




### Hyperparameters

- The network consists of 2 fully connected hidden layers with 128 and 32 neurons respectively
- Both layers use ReLu activation functions
- Replay buffer size: `BUFFER_SIZE = int(1e5)`
- Minibatch size: `BATCH_SIZE = 128`
- Discount factor: `GAMMA = 0.99`
- Soft update factor of target parameters: `TAU = 1e-3`
- Learning rate: `LR = 5e-4`
- Network update frequency: `UPDATE_EVERY = 4`
- epsilon starting value: `eps_start = 1.0`
- epsilon min value: `eps_min = 0.001`
- epsilon decay rate: `eps_decay = 0.95`


### Running

To start training enter on the command line:
```sh
$ python main.py
```

The training should be completed after 224 episodes with the final average score of +13.03.
After the training is completed a file `weights.pt` is created automatically that contains the weights of the trained network.
Also a scores plot is created in the file `scores.png`.


### Ideas for Future Work

The neural network can be improved by implementing the following:
- Double DQN (code is inserted but  commented out)
- Dueling DQN
- Prioritized Experience Replay
- A3C 


