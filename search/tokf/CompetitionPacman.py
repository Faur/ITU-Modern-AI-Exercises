from game import Agent
from game import Directions

class CompAgent(Agent):
    def __init__(self, **args):
        Agent.__init__(self, **args)


    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."

        """ YOUR CODE HERE! """
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP
