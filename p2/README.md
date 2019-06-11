[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control

### Introduction

This is a deep reinforcement learning project to train a single agent to 
move a double-jointed arm to a target location and
maintain its position at the target location.
The project is implemented with [Python 3.6](https://www.python.org/downloads/release/python-360/), [PyTorch](https://pytorch.org/), [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

For the purpose of the project the Unity ML [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment is used.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.
Each action is a vector with four numbers, corresponding to torque applicable to two joints.
Every entry in the action vector is a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Getting Started

1. Download the Unity environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): c[lick here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the `src` folder and unzip (or decompress) it. 

### Instructions

##### 1. Start the Environment

Open the file `src/main.py` with a python editor and verify that
`file_name` parameter matches the location of the Unity environment that was downloaded and unzipped:
```python
env = UnityEnvironment(file_name="./Reacher_Linux_NoVis/Reacher.x86_64")
```

##### 2. Train the Agent
Change folder to `src` and start training the agent:
```sh
$ python main.py
```
The training is completed when the agent achieves an average score of +30 over 100 consecutive episodes.

As a result, the program automatically saves the trained network weights into `weights.pt` and plots earned scores as `scores.png`.

===

The application has been tested on AWS using instance type `p2.xlarge` and image `Deep Learning AMI (Ubuntu) Version 23.0`.