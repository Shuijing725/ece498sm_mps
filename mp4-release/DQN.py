
# coding: utf-8

# In[1]:


# !pip install cmake 'gym[atari]' scipy numpy torch


# In[4]:


#Understand the **Taxi-v2** environment in gym. We first create the environment
import gym
import numpy as np
import random
from collections import deque
from IPython.display import clear_output

import math
import torch
import numpy as np

print ("modules imported!")


# In[7]:


#Create the taxi-v2 environment
env = gym.make("Taxi-v2").env

env.render() #creates the simulation in python


# In[8]:


env.reset() # reset environment to a new, random state
env.render()

print("Action Space: ", env.action_space) #action space of the Taxi-environment
print("State Space: ", env.observation_space)


# Let's first understand the state-space of this problem. We define the state of the taxi with the following co-ordinates 
# 
# (row, column, passenger_index, destination_index)
# 
# where, *row* and *column* denotes the location of the taxi in a 5X5 environment. 
# *passenger_index* : is the index number of passenger in the taxi. It could either be empty(0) or filled with either one of the passengers(1,2,3,4). Similarly, destination index is one of the four locations of the taxi(1,2,3,4) for one of the four locations. Thus, the total possible state space of the problem is as follows, 5X5X5X4 = 500
# 
# The action space of this algorithm has six values. Of these the first four values denote the direction of car's going. The next two action space is to either pick-up or drop-off the car at the particular location. 

# In[9]:


state_e = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
state_d = env.decode(state_e)
env.render()
print ("encoded state", state_e)
print ("decoded_state", np.array(list(state_d)))


# In[10]:


'''
Let's now, run a policy wherein the actions are randomly choosen
to get an idea how good or worse a random policy performs and the 
'''
# env.reset()  # set environment to illustration's state
#
# epochs = 0
# penalties, reward = 0, 0
#
# frames = [] # for animation
#
# done = False
#
# while not done:
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)
#
#     if reward == -10:
#         penalties += 1
#
#     # Put each rendered frame into dict for animation
#     frames.append({
#         'frame': env.render(mode='ansi'),
#         'state': state,
#         'action': action,
#         'reward': reward
#         }
#     )
#
#     epochs += 1
#
#
# print("Timesteps taken:", format(epochs))
# print("Penalties incurred:",format(penalties))


# # Q-learning Algorithm
# In the next cell. You are required to implement the Q-learning algorithm. 
# 
# This has been described in MP4 write up as well as in class. 
# 

# In[6]:


# Q-learning algorithm implementation
# Set Hyper-parameters for the Q-learning algorithm
# alpha = 0.1 # learning rate
# gamma = 0.6
# epsilon = 0.1 #epsilon value for epsilon-greedy policy
# # Initialise the Q-table for the problem
# q_table = np.zeros([env.observation_space.n, env.action_space.n])
# # Survelliance data for the whole run of this algorithm
# all_epochs = []
# all_penalties = []
# M = 100001 # episodic runs of the Q-learning algorithm
# for i in range(1, M):
#     state = env.reset()
#     epochs, penalties, reward, = 0, 0, 0
#     done = False
#     while not done:
#         ###
#         #Add you code here.
#         #Remember action in line 24 would be based on the epsilon-greedy policy
#         #described above
#         ###
#         random_num = np.random.uniform(0, 1)
#         if random_num < epsilon:
#             action = env.action_space.sample()
#         else:
#             action = np.argmax(q_table[state])
#         next_state, reward, done, info = env.step(action) #next state
#         #env.render() #render the simulation to see it on the screen
#
#
#         ###
#         # Add the remaining part of the code here
#         # Update Q table after taking action above
#         ###
#         old_value = q_table[state, action]
#         next_max = np.max(q_table[next_state])
#         q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
#
#
#         if reward == -10:
#             penalties += 1
#
#         if epochs > 100:
#             break
#
#         state = next_state
#         epochs += 1
#
#     if i % 100 == 0:
#         clear_output(wait=True)
#         print("Episode:", i)
#
# print("Training finished.\n")


# ## Test
# Test your Q-learning implementation in the cell below

# In[7]:


# Testing Q-learning implementation
# def test(q_table, episodes = 100):
#     total_epochs, total_penalties = 0, 0
#     #episodes = 100
#
#     for _ in range(episodes):
#         state = env.reset()
#         epochs, penalties, reward = 0, 0, 0
#         done = False
#
#         while not done:
#             action = np.argmax(q_table[state])
#             state, reward, done, info = env.step(action)
#             #print (state)
#             #env.render()
#
#             if reward == -10:
#                 penalties += 1
#
#             epochs += 1
#
#             if epochs > 100:
#                 break
#
#         total_penalties += penalties
#         total_epochs += epochs
#
#     print("Results after {episodes} episodes:")
#     print("Average timesteps per episode:", total_epochs/episodes)
#     print("Average penalties per episode:", total_penalties/episodes)
# test(q_table)


# # Towards Function Approximations in Reinforcement learning
# You will observe that training a simple environment with state, action pairs of 500×6 elements takes a long time to converge using Q-learning. Additionally, you have to create 500×6 element size Q-table to store all the Q-values. This process of making such a large Q-table is memory intensive. You also may have noticed that this method of using a Q-table only works when the state and action spaces is discrete. These practical issues with Q-learning was what originally prevented it being from used in many real-world scenarios.
# 
# If the state space/action spaces are large, an alternative idea is to approximate the Q-table with a function. Suppose you have a state space of around 106 states and action space is continuous value from [0,1]. Instead of trying to represent the Q table explicitly, we can approximate the table with a function.  While function approximation is almost as old as theory of RL itself, it recently regained popularity due to a seminal paper on a Neural Network based Q-learning algorithm called the DQN algorithm, which was proposed in 2015.
# 
# In this MP, you have been asked to implement the Deep Q-learning algorithm (DQN).  The idea of this algorithm is to learn a sufficiently close approximation to the optimal Q-function using a Neural Network architecture.
# 

# # DQN architecture
# In the cell below is the Neural Network architecture for Deep Q-learning for this MP. You don't have to implement anything here but please go through the code

# In[11]:


import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

class QFunction(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate, epsilon, seed, batch_size, tau, duel_enable = False, duel_type = 'avg'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate 
        self.epsilon = epsilon
        self.seed = seed
        self.batch_size = batch_size
        self.tau = tau
        super(QFunction, self).__init__()
        # Define the two layered network
        self.layer1 = nn.Linear(self.state_dim,48)
        n = weight_init._calculate_fan_in_and_fan_out(self.layer1.weight)[0]
        torch.manual_seed(self.seed)
        self.layer1.weight.data.uniform_(-math.sqrt(6./n), math.sqrt(6./n))
        
        self.layer2 = nn.Linear(48,action_dim)
        n = weight_init._calculate_fan_in_and_fan_out(self.layer2.weight)[0]
        torch.manual_seed(self.seed)
        self.layer2.weight.data.uniform_(-math.sqrt(6./n), math.sqrt(6./n))
        
        # Define the loss function and the optimizer that is being used
        self.loss_fn = torch.nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = 0.01)

    def forward(self, x):
        y = F.relu(self.layer1(x))
        y = self.layer2(y)
        return y


    def train(self, states, actions, y):
        self.optimizer.zero_grad()
        q_value = self.forward(states)
        actions = actions.data.numpy().astype(int)
        range_array = np.array(range(self.batch_size))
        index_range = np.arange(self.batch_size)
        index_range = np.reshape(index_range,(1, self.batch_size))
        q_value = q_value[index_range, actions]
        loss = self.loss_fn(q_value,y)
        # print('loss:', loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_target_weights(self, critic):

        for weight,target_weight in zip(self.parameters(),critic.parameters()):
            weight.data = (1-self.tau)*weight.data +  (self.tau)*target_weight.data


# # Replay Buffer
# In the next cell we have defined replay buffer. Again, you don't have to implement anything here but please go through the code for the buffer because it is used in almost all Deep Reinforcement learning applications

# In[12]:


#code for the ReplayBuffer
import numpy as np
import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=None):
        """
        The right side of the deque contains the most recent experiences
        The buffer stores a number of past experiences to stochastically sample from
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=self.buffer_size)
        self.seed = random_seed
        if self.seed is not None:
            random.seed(self.seed)

    def add(self, state, action, reward, t, s2):
        experience = (state, action, reward, t, s2)
        self.buffer.append(experience)
        self.count += 1

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(batch_size, -1)
        t_batch = np.array([_[3] for _ in batch]).reshape(batch_size, -1)
        s2_batch = np.array([_[4] for _ in batch])
        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


# # DQN
# In the next cell; you are going to implement the DQN learning algorithm. Specifically, you are implementing the epsilon-greedy policy and the weight update parts of the code. 

# In[ ]:


import os
import gym
import sys
import math
import copy
import random
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from gym import wrappers
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable


class DDQNAgent:
    def __init__(self, critic, replay_buffer, episode_len=1000, episode_steps=1000, epsilon=0.01, epsilon_decay=0.999,
                 batch_size=64, gamma=0.99, seed=1234):
        self.critic = copy.deepcopy(critic)
        self.target_critic = copy.deepcopy(critic)
        self.replay_buffer = copy.deepcopy(replay_buffer)
        self.episode_len = episode_len
        self.episode_steps = episode_steps
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.seed = seed

    def take_action(self, state, n_actions, epsilon_t):
        random_num = np.random.uniform(0, 1)
        ##########
        # epsilon_t = 0
        ##########
        if random_num < epsilon_t:
            # act randomly
            return env.action_space.sample()
        else:
            values = self.critic.forward(state)
            values = values.cpu().detach().numpy()  # values = [[float, float, ..., float]]
            return np.argmax(values[0])

    def train(self, env):
        CUDA = torch.cuda.is_available()
        epsilon_t = 1.0
        for i in range(self.episode_len):
            j = 0
            state = time()
            env.seed(self.seed + i)
            s = env.reset()
            terminal = False
            clear_output(wait=True)
            print("iterations", i)
            while not terminal:

                j = j + 1
                input_state = np.reshape(s, (1, self.critic.state_dim))
                input_state = torch.from_numpy(input_state)
                dtype = torch.FloatTensor
                input_state = Variable(input_state.type(dtype), requires_grad=True)
                if CUDA:
                    input_state = input_state.cuda()

                a = self.take_action(input_state, env.action_space.n, epsilon_t)
                # print('action taken:', a)
                s2, r, terminal, info = env.step(a)

                self.replay_buffer.add(np.reshape(s, (self.critic.state_dim,)),
                                       a,
                                       r, terminal, np.reshape(s2, (self.critic.state_dim,)))
                # epsilon-greedy policy
                if epsilon_t > self.epsilon:
                    epsilon_t = epsilon_t * self.epsilon_decay
                else:
                    epsilon_t = self.epsilon
                if self.replay_buffer.size() > self.batch_size:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(self.batch_size)

                    s2_batch = torch.from_numpy(s2_batch)
                    s2_batch = Variable(s2_batch.type(torch.FloatTensor), requires_grad=False)

                    s_batch = torch.from_numpy(s_batch).type(torch.FloatTensor)
                    a_batch = torch.from_numpy(a_batch).type(torch.FloatTensor)
                    r_batch = torch.from_numpy(r_batch).type(torch.FloatTensor)

                    # find y:
                    q_values = self.target_critic.forward(s2_batch)
                    maxQ = torch.max(q_values, 1)[0]
                    maxQ = torch.reshape(maxQ, r_batch.shape)
                    assert (r_batch.shape == maxQ.shape)
                    y = r_batch + self.gamma * maxQ

                    # print(t_batch)
                    for i in range(len(y[0])):
                        if t_batch[i, 0]:
                            y[i, 0] = r_batch[i, 0]


                    if CUDA:
                        s2_batch = s2_batch.cuda()
                        s_batch = s_batch.cuda()
                        a_batch = a_batch.cuda()
                        y = y.cuda()
                    ######################
                    # Add your code here
                    # You have to write the
                    # gradient step of the Q-Function here.
                    #####################

                    self.critic.train(s_batch, a_batch, y)
                    if i % 1 == 0:
                        self.target_critic.update_target_weights(self.critic)

                else:
                    loss = 0
                if j > 100:
                    terminal = True

                s = s2

    def test(self, env):
        print ('##########Test begins!#########')
        total_epochs, total_penalties = 0, 0
        episodes = 1
        total_reward = 0
        CUDA = torch.cuda.is_available()

        for i in range(episodes):
            state = time()
            s = env.reset()
            epochs, penalties, reward = 0, 0, 0
            print("iterations", i)
            j = 0
            terminal = False
            while not terminal:
                env.render()
                j += 1
                input_state = np.reshape(s, (1, self.critic.state_dim))
                input_state = torch.from_numpy(input_state)
                dtype = torch.FloatTensor
                input_state = Variable(input_state.type(dtype), requires_grad=True)

                if CUDA:
                    input_state = input_state.cuda()

                a = self.critic(input_state)
                a = a.data.cpu().numpy()
                a = np.argmax(a)
                s, r, terminal, info = env.step(a)
                total_reward += r
                if r == 20:
                    print("dropped off")
                if r == -10:
                    penalties += 1
                epochs += 1
                if terminal or j > 100:
                    break

            total_penalties += penalties
            total_epochs += epochs

        print("Results after {episodes} episodes:")
        print("Total reward: ", total_reward)
        print("Average timesteps per episode:", total_epochs / episodes)
        print("Average penalties per episode:", total_penalties / episodes)


# In[ ]:


class arg:
    def __init__(self):
        self.seed = 1234
        self.tau = 0.001
        self.learning_rate =0.01
        self.batch_size=64
        self.bufferlength=2000
        self.l2_decay=0.01
        self.gamma=0.6
        self.episode_len=500
        self.episode_steps=1000
        self.epsilon=0.1
        self.epsilon_decay=0.999
        self.is_train=True
        self.actor_weights='ddqn_cartpole'


# # Test DQN implementation
# Test your DQN impelementation. It first trains the Q-networks and then proceeds to test the implementation

# In[ ]:


import os
import sys
import gym
import math
import copy
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from time import time
from gym import wrappers
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as weight_init

def main(args):
    CUDA = torch.cuda.is_available()
    
    env = gym.make("Taxi-v2").env
    state_dim = 1
    action_dim = 6

    qfunction = QFunction(state_dim, action_dim, learning_rate = args.learning_rate, epsilon = args.epsilon, seed = args.seed, batch_size = args.batch_size, tau = args.tau)

    if CUDA: 
        qfunction = qfunction.cuda()

    replay_buffer = ReplayBuffer(args.bufferlength)

    agent = DDQNAgent(qfunction, replay_buffer, episode_len = args.episode_len,
                     episode_steps=args.episode_steps, epsilon = args.epsilon, epsilon_decay = args.epsilon_decay,
                     batch_size = args.batch_size, gamma = args.gamma, seed = args.seed)

    if args.is_train:
        agent.train(env)
        agent.test(env)


if __name__ == '__main__':
    args = arg()
    main(args)

