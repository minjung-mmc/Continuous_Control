[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Report for Project 2: Continuous Control for 20 Agents
    
## Introduction

In this project an agent (or agents) aims to follow a target. The goal of agents is to maintain its position at the target location for as many time steps as possible. you can see an implementation for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

![Trained Agent][image1]
## Implementation specs

### 1. Summary

I implement the [Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1509.02971)

--------

### 2. Details

#### 2-1 Concepts

DDPG is composed of two networks: one actor and one critic.
During a step, the actor is used to estimate the best action. the critic then use this value as in a DDQN to evaluate the optimal action value function.
Both of the actor and the critic are composed of two networks: Local Network and Target Network. 
During the training, the actor is updated by applying the chain rule to the expexted return from the start distribution. The critic is updated by comparing the expected return of the current state to the sum of the reward of the chosen action and the expected return of the next state.


##### 2-1-1 Learning Algorithm

1. Initialize the replay memory, local Actor and local Critic Network.
2. Actor Network does the policy approximation while Critic Netrowk does value estimation. 
3. Copy generated weights to the target Actor and target Critic Network every iteration.
4. Train the agents for some episodes. Traing loop is composed of two steps: acting and learning.
5. Update the target Actor and the target Critic weights by making a copy of the current weights of the local Actor and Critic Network.

------------
#### 2-2. Networks

The network structure is as follows:

##### 2-2-1. Actor

state -> BatchNorm -> Linear(state_size, 256) -> BatchNorm -> LeakyRelu -> Linear(256, 128) -> BatchNorm -> LeakyRelu -> Linear(128, action_size) -> tanh

##### 2-2-2. Critic

state -> BatchNorm -> Linear(state_size, 256) -> Relu -> (concat with action) -> Linear(256+action_size, 128) -> Relu -> Linear(128, 1) 

#### 2-3. Hyperparameters

Agent hyperparameters are passed as constructor arguments to `Agent`.  The default values, used in this project, are:

| parameter    | value  | description                                                                   |
|--------------|--------|-------------------------------------------------------------------------------|
| BUFFER_SIZE  | 1e5    | Number of experiences to keep on the replay memory for the TD3                |
| BATCH_SIZE   | 128    | Minibatch size used at each learning step                                     |
| GAMMA        | 0.99   | Discount applied to future rewards                                            |
| TAU          | 1e-3   | Scaling parameter applied to soft update                                      |
| LR_ACTOR     | 1e-3   | Learning rate for actor used for the Adam optimizer                           |
| LR_CRITIC    | 1e-3   | Learning rate for critic used for the Adam optimizer                          |
| WEIGHT_DECAY | 0      | L2 weight decay                                                               |


Training hyperparameters are passed to the training function `train` of `Agent`, defined below.  The default values are:

| parameter                     | value            | description                                                             |
|-------------------------------|------------------|-------------------------------------------------------------------------|
| n_episodes                    | 5000             | Maximum number of training episodes                                     |
| max_t                         | 1000             | Maximum number of steps per episode                                     |


-----------

### 3. Result and Future works

#### 3-1. Reward


Here x-axis is the episode and y-axis is the reward.

#### 3-2. Future works

1. Parameters tuning for DDPG. 
2. Implement this project by other algorithms like **PPO(Proximal Policy Optimization)** which is on-policy algorithm or **SAC(Soft Actor Critic)** which is off policy with entropy maximization to enable stability and exploration.
   