from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Cliff():
    def __init__(self):
        self.valid_actions = range(4) # Up, down, left, right
        self.num_y = 4
        self.num_x = 12
        self.start = np.array([0, 0])
        self.terminal = np.array([self.num_x - 1, 0])
        self.cliff = np.array([[i, 0] for i in range(1, self.num_x - 1)])

        self.heat_map = np.zeros([self.num_x, self.num_y])

    def reset(self):
        self.pos = np.copy(self.start)
        self.heat_map[tuple(self.pos)] += 1
        return np.copy(self.pos), 0, False, None

    @property
    def state_shape(self):
        return np.array([self.num_x, self.num_y])

    @property
    def num_actions(self):
        return len(self.valid_actions)

    def _check_pos(self):
        if (self.pos == self.terminal).all():  # Terminate
            r = 0
            d = True
            self.heat_map[tuple(self.pos)] += 1
        elif self.pos[0] in self.cliff[:, 0] and self.pos[1] in self.cliff[:, 1]:  # Fall of cliff
            r = -100
            d = False  # cliff shouldn't terminate
            self.heat_map[tuple(self.pos)] += 1
            self.reset()
        else:
            r = -1
            d = False
        return r, d

    def render(self):
        img = np.zeros([self.num_x, self.num_y])
        img[range(1, self.num_x - 1), 0] = 0.5
        img[tuple(self.pos)] = 1
        img[self.num_x - 1, 0] = 1
        return img

    def random_action(self):
        return np.random.randint(len(self.valid_actions))

    def step(self, action):
        assert action in self.valid_actions, 'Invalid action: ' + str(action)

        self.heat_map[tuple(self.pos)] += 1
        if action == 0 and self.pos[0] > 0:  # 0=Left
            self.pos[0] -= 1
        elif action == 1 and self.pos[1] < self.num_y - 1:  # 1=Up
            self.pos[1] += 1
        elif action == 2 and self.pos[0] < self.num_x - 1:  # 2=right
            self.pos[0] += 1
        elif action == 3 and self.pos[1] > 0:  # 3=down
            self.pos[1] -= 1

        r, d = self._check_pos()
        return np.copy(self.pos), r, d, None
