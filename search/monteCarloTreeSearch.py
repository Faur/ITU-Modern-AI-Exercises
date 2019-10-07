import copy

from pacman import *
from game import Agent


class MCTSagent(Agent):
    def __init__(self):
        self.explored = {}  # Dictionary for storing the explored states
        self.n = 10  # Depth of search  # TODO: Play with this once the code runs
        self.c = 1  # Exploration parameter # TODO: Play with this once the code runs

    def getAction(self, state):
        """ Main function for the Monte Carlo Tree Search. For as long as there
            are resources, run the main loop. Once the resources runs out, take the
            action that looks the best.
        """
        self.explored = {}

        root = """ YOUR CODE HERE"""  # TODO: How will you encode the nodes and states?

        for _ in range(self.n):  # while resources are left (time, computational power, etc)
            leaf_list = self.traverse(root)
            for leaf in leaf_list:
                simulation_result = self.rollout(leaf)
                self.backpropagate(leaf, simulation_result)

        return self.best_action(root)

    def all_successors(self, state):
        """ Returns all legal successor states."""
        next_pos = []
        for action in state.getLegalPacmanActions():
            next_pos.append(state.generatePacmanSuccessor(action))
        return next_pos

    def traverse(self, state):
        """ Returns a list of states to explore. If state is terminal the list
            has length 1.
        """

        def state_is_explored(state):
            """ Determines whether a state has been explored before.
                Returns True if the state has been explored, false otherwise
            """
            """ YOUR CODE HERE!"""
            raise NotImplementedError

        def best_UCT(state):
            """ Given a state, return the best action according to the UCT criterion."""
            """ YOUR CODE HERE!"""
            raise NotImplementedError

        while state_is_explored(state):
            action = best_UCT(state)
            state = state.generatePacmanSuccessor(action)

            if state.isWin() or state.isLose():
                return [copy.deepcopy(state)]

        return self.all_successors(state)

    def rollout(self, state):
        """ Simulate a play through, using random actions.
        """
        while not state.isWin() and not state.isLose():
            """ YOUR CODE HERE! """
            state = 'XXX'
            raise NotImplementedError
        return state.getScore()

    def backpropagate(self, state, result):
        """ Backpropagate the scores, and update the value estimates."""
        """ YOUR CODE HERE! """

    def best_action(self, node):
        """ Returns the best action given a state. This will be the action with
            the highest number of visits.
        """
        """ Your Code HERE!"""
        action = None
        return action


if __name__ == '__main__':
    str_args = ['-l', 'TinyMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    str_args = ['-l', 'TestMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    args = readCommand(str_args)
    # args['display'] = textDisplay.NullGraphics()  # Disable rendering

    args['pacman'] = MCTSagent()
    out = runGames( **args)

    scores = [o.state.getScore() for o in out]
    print(scores)
