"""
Code for ITU Modern AI course is based on tutorial by Morvan Zhou
    https://github.com/MorvanZhou/PyTorch-Tutorial
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

# Initialize openAI gym environment
env = gym.make('CartPole-v1')
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

# Define network parameters
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

# Define Training Parameters
""" YOUR CODE HERE!
    You can and should play around with these!
    But these are reasonable first guesses.
"""
BATCH_SIZE = 32
LR = 0.001                   # learning rate
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 100    # target update frequency
MEMORY_CAPACITY = 10000      # Size of experience memory buffer

# Define epsilon-greedy parameters. We want to start with very random
# and slowly use more and more deterministic actions
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.999


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        # Initialize weights
        # nn.Linear is a class that applies a linear transformation to the incoming data
        # i.e. it manages the weights and biases for us
        # init takes two parameters:
        #   in_features: size of each input sample
        #   out_features: size of each output sample

        """ YOUR CODE HERE!
            Code below is intentionally bad. You should fix it up :)            
            Make a network that solves the CartPole environment.
            Remember to use proper initialization e.g. Glorot (see Lab 5)
            PyTorch can help with that, so have a Google :-)
            
            [You will define the activation functions in self.forward below]
        """
        fc1_units = 1
        self.fc1 = nn.Linear(N_STATES, fc1_units)
        self.out = nn.Linear(fc1_units, N_ACTIONS)

        fc1_init_std = 0.1
        out_init_std = 0.1

        self.fc1.weight.data.normal_(0, fc1_init_std)   # initialization
        self.out.weight.data.normal_(0, out_init_std)   # initialization

    def forward(self, x):
        """ Computes the forward pass of the network, and
            and returns the Q values for each action
        """

        """ YOUR CODE HERE!
            Complete this function, based on the network architecture
            that you have chosen in Net.__init__
        """
        # Hidden layer
        x = F.relu(self.fc1(x))

        # Output layer
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))     # initialize memory
        self.loss_func = nn.MSELoss()

        # PyTorch provides advanced optimizers, that improve upon the
        # basic stochastic gradient descent.
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

    @property
    def eps(self):
        return np.max([EXPLORATION_MIN, EXPLORATION_MAX*EXPLORATION_DECAY**self.learn_step_counter])

    def choose_action(self, x):
        """ Return an action given a state x."""
        # Convert the numpy array into a Tensor of appropriate shape.
        # The returned tensor shares the same underlying data (refrence)
        # with this tensor.
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if np.random.uniform() < self.eps:
            # Random Action
            action = env.action_space.sample()
        else:
            # Greedy Action
            # Compute the Q value using the network
            actions_value = self.eval_net.forward(x)

            # Select the action with the highest value
            action = torch.max(actions_value, 1)[1]

            # Converting back and forth between numpy is easy!
            action = action.data.numpy()

            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_, done):
        """ Store transition, replacing the old memory with new memory
            once the memory bank is full.
        """
        transition = np.hstack((s, [a, r], s_, done))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Update target network once every TARGET_REPLACE_ITER updates
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE, replace=False)
        b_memory = self.memory[sample_index, :]

        # Convert into tensors
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, N_STATES+2:-1])
        b_done = torch.FloatTensor(b_memory[:, -1])

        # Get the current estimate of the Q value based on the state-action pair
        # from memory.
        q_eval = self.eval_net(b_s)  # shape (batch, 1)

        # Gather returns values along an axis specified by dim.
        q_eval = q_eval.gather(1, b_a)  # shape (batch, 1)

        # Compute the Q value using the target network
        q_next = self.target_net(b_s_)

        # We don't want to backpropagate the errors to the target network,
        # so we detach it from the graph
        q_next = q_next.detach()

        # Select the highest Q value, and discount it
        q_next = GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        # When the episode is done we don't want to bootstrap the Q value
        # Remove (zero) in the instances where the episode ended
        q_next = q_next * (1-b_done.unsqueeze(-1))

        q_target = b_r + q_next
        loss = self.loss_func(q_eval, q_target)

        # We must zero the gradients manually every time
        self.optimizer.zero_grad()

        # Compute the gradients
        loss.backward()

        # Update the weights
        self.optimizer.step()


dqn = DQN()

# reward_shaping = True
reward_shaping = False

print('\nCollecting experience...')
episode_rewards = []
training_begins = None
should_render = False
i_episode = 0
while True:
    i_episode += 1
    s = env.reset()
    ep_r = 0
    while True:
        if should_render:
            env.render()  # comment out if you don't want to see the render (run faster)
        a = dqn.choose_action(s)

        # take action, and take one step in the environment
        s_, r, done, info = env.step(a)
        ep_r += r

        if reward_shaping:
            # Modify the reward so that the problem becomes easier to solve.
            # Reward shaping can have a huge impact on how difficult a problm is.
            # Here we access parts of the environment that normally aren't available
            # through env.unwrapped.
            # Here we modify the reward to include information about how close to the
            # center the Cart is, and how upright the pole is, both normalized to [0,1]

            """ YOUR CODE HERE!
                Implement reward shaping.
                See https://github.com/openai/gym/wiki/CartPole-v0 for details on CartPole
                
                Inspiration: 
                    x, x_dot, theta, theta_dot = s_
                    x_thresh = env.unwrapped.x_threshold
                    theta_thresh = env.unwrapped.theta_threshold_radians
            """
            pass

        # Store experience
        dqn.store_transition(s, a, r, s_, done)

        # Normally we don't update the network before the experience memory buffer is full.
        if dqn.memory_counter >= MEMORY_CAPACITY:
            if training_begins is None:
                # should_render = True  # Set true if you want to see performance
                training_begins = i_episode

            dqn.learn()

            if done:
                print('Ep: {:6}'.format(i_episode),
                      '| Ep_r: {:6}'.format(round(ep_r, 2)),
                      '| eps: {:5.2f}'.format(dqn.eps)
                      )
                episode_rewards.append(ep_r)

        if done:
            episode_rewards.append(ep_r)
            break
        s = s_

plt.clf()
plt.plot(range(len(episode_rewards)), episode_rewards, label='Training Begun')
plt.plot(range(len(episode_rewards[:training_begins])), episode_rewards[:training_begins], label='Gathering Exp.')
plt.legend()

print(":)")