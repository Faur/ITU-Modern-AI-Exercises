import numpy as np
import matplotlib.pyplot as plt

from game import Directions
from game import Agent

import NN_util


def init_NN_Glorot(L, activations, uniform=False):
    """
    Initializer using the glorot initialization scheme
    """

    weights = []
    biases = []
    """ YOUR CODE HERE!"""

    return (weights, biases)



def backward_pass(x, t, y, z, a, NN, activations, loss):
    """
    This function performs a backward pass ITERATIVELY. It saves lists all of the derivatives in the process

    Input:
    x:           The input used for the batch                (np.array)
    t:           The observed targets                        (np.array, the first dimension must be the same to x)
    y:           The output of the forward_pass of NN for x  (np.array, must have the same shape as t)
    a:           The affine transforms from the forward_pass (np.array)
    z:           The activated units from the forward_pass (np.array)
    activations: The activations to be used                  (list of functions)
    loss:        The loss function to be used                (one function)

    Output:
    g_w: A list of gradients for every hidden unit
    g_b: A list of gradients for every bias

    Shapes for the einsum:
    b: batch size
    i: size of the input hidden layer (layer l)
    o: size of the output (layer l+1)
    """
    BS = x.shape[0]  # Implied batch shape

    # First, let's compute the list of derivatives of z with respect to a
    d_a = []
    for i in range(len(activations)):
        d_a.append(activations[i](a[i], derivative=True))

    # Second, let's compute the derivative of the loss function
    t = t.reshape(BS, -1)

    """ YOUR CODE HERE!"""
class DQNagent(Agent):
    def __init__(self, **args):
        """ Initialize the neural network etc."""
        """ DON'T CHANGE THIS PART"""
        Agent.__init__(self, **args)
        self.verbose = False

        self.a_dict = NN_util.TwoWayDict()
        self.a_dict["North"] = 0
        self.a_dict["East"] = 1
        self.a_dict["South"] = 2
        self.a_dict["West"] = 3
        # self.a_dict["Stop"] = 4
        self.num_actions = len(self.a_dict)

        self.prev_state = None
        self.prev_action = None
        self.prev_score = 0.0
        self.exp = []

        """ PLAY AROUND AND CHANGE THIS PART"""
        self.n = 1  # For n-step returns
        self.eps = 0.1  # For epsilon greedy action selection.
        self.alpha = 0.01  # learning rate
        self.gamma = 0.99  # discount factor

        self.layers = [2, 32, self.num_actions]
        self.activation_fns = [NN_util.ReLU, NN_util.Linear]
        assert len(self.layers) == len(self.activation_fns) + 1, "Number of layers and activation functions don't match!"

        """ DON'T CHANGE THIS PART"""
        # try:
        #     # Load weights if they exist
        #     self.NN = self.load_weights()
        # except IOError:
        self.NN = init_NN_Glorot(self.layers, self.activation_fns)

    def compute_G(self):
        """ Returns the discounted reward. """
        G = 0
        for i in range(self.n):
            G = G + (self.gamma ** i) * self.exp[i-self.n][2]
        return G

    def getNetworkInput(self, state):
        return np.asarray([state.getPacmanPosition()])

    def getReward(self, state):
        r = state.getScore() - self.prev_score
        self.prev_score = state.getScore()
        return r

    def updateNetwork(self, n, terminal=False):
    def terminal_update(self, state):
        s = self.getNetworkInput(state)
        r = self.getReward(state)

        for i in range(self.n):
            self.updateNetwork(self.n-i, terminal=True)

        self.save_weights()

        ## Clear everything, ready for next run!
        self.prev_state = None
        self.prev_action = None
        self.prev_score = 0.0
        self.exp = []


    def getAction(self, state):
        """
        """

        s = self.getNetworkInput(state)
        r = self.getReward(state)

        if np.random.uniform() > self.eps:
            NN_forward = NN_util.forward_pass(s, self.NN, self.activation_fns)
            a = np.argmax(NN_forward[1][-1])

            if self.a_dict[a] not in state.getLegalActions():
                # If illegal action is selected - set a large negative reward
                # We also force the action to be 'Stop', but we do that in the
                # end of the function.
                r = -50
                if self.verbose:
                    print(s, 'illegal', self.a_dict[a], NN_forward[1][-1])
            else:
                if self.verbose:
                    print(s, 'pick', self.a_dict[a], NN_forward[1][-1])

        else:
            # Pick random legal action
            legal_actions = state.getLegalActions()
            legal_actions.remove("Stop")
            a = self.a_dict[np.random.choice(legal_actions)]
            if self.verbose:
                print(s, 'random', self.a_dict[a])


        if self.prev_state is not None:
            self.exp.append([self.prev_state, self.prev_action, r, s, a])

            self.updateNetwork(self.n)

        self.prev_state = s
        self.prev_action = a

        if self.a_dict[a] not in state.getLegalActions():
            return "Stop"

        return self.a_dict[a]

    def save_weights(self):
        # TODO
        pass

    def load_weights(self):
        # TODO
        pass



