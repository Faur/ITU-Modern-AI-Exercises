from collections import defaultdict, namedtuple
from random import choice

class Node():
    """
    A representation of a single game state.
    MCTS works by constructing a tree of these Nodes.
    Could be any game e.g. a chess or checkers board state.
    """
    def __init__(self, state):
        self.state = state

    def find_children(self):
        """ All possible successors of this game state.
            Returns a set
        """
        children = set()
        for action in self.state.getLegalPacmanActions():
            children.add(Node(self.state.generatePacmanSuccessor(action)))
        return children

    def find_random_child(self):
        """ Random successor of this board state
            (for more efficient simulation)
            Returns a Node
        """
        action = choice(self.state.getLegalPacmanActions())
        return Node(self.state.generatePacmanSuccessor(action))

    def is_terminal(self):
        """ Returns True if the node has no children"
            Returns a boolean
        """
        return self.state.isLose() or self.state.isWin()

    def reward(self):
        """ Assumes `self` is terminal node.
            returns a number (int or float)
        """
        return self.state.getScore()

    def __hash__(self):
        """ Nodes must be hashable"
            returns a hash that should be unique for each game state
        """
        return self.state.__hash__()

    def __eq__(node1, node2):
        """ Nodes must be comparable
            returns a boolean
        """
        return node1.state.__hash__() == node2.state.__hash__()


###################
### TIC TAC TOE ###
###################

def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal


def _find_winner(tup):
    "Returns None if no winner, True if X wins, False if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None


_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")
# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class TicTacToeBoard(_TTTB):
# class TicTacToeBoard(_TTTB, Node):
    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(i) for i, value in enumerate(board.tup) if value is None
        }

    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value is None]
        return board.make_move(choice(empty_spots))

    def reward(board):
        if not board.terminal:
            raise RuntimeError("reward called on nonterminal board {board}")
        if board.winner is board.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError("reward called on unreachable board {board}")
        if board.turn is (not board.winner):
            return 0  # Your opponent has just won. Bad.
        if board.winner is None:
            return 0.5  # Board is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError("board has unknown winner type {board.winner}")

    def is_terminal(board):
        return board.terminal

    def make_move(board, index):
        tup = board.tup[:index] + (board.turn,) + board.tup[index + 1 :]
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        return TicTacToeBoard(tup, turn, winner, is_terminal)

    def to_pretty_string(board):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )


def new_tic_tac_toe_board():
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)


def play_game(tree, human=False, deterministic=True):
    board = new_tic_tac_toe_board()

    while True:
        if human:  # Human
            print(board.to_pretty_string())
            valid_move = False
            while not valid_move:
                row_col = input("enter row,col: ")
                # row, col = map(int, row_col.split(","))  # python 3
                row, col = row_col  # python 2
                index = 3 * (row - 1) + (col - 1)
                if board.tup[index] is not None:
                    print("Invalid move")
                    pass
                else:
                    valid_move = True
            board = board.make_move(index)
            print(board.to_pretty_string())
        else:  # Random moves
            valid_move = False
            index = -1
            while not valid_move:
                if deterministic:
                    index += 1
                else:
                    index = choice([0,1,2,3,4,5,6,7,8])
                if board.tup[index] is not None:
                    pass
                else:
                    valid_move = True
            # print(board.to_pretty_string())
            board = board.make_move(index)
            # print(board.to_pretty_string())

        if board.terminal:
            break

        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(100):
            tree.do_rollout(board)
        board = tree.choose(board)
        if board.terminal:
            break

    print(board.to_pretty_string())
    winner = _find_winner(board.tup)
    if winner == True:
        print('You win!')
        return 1
    elif winner == False:
        print('MCTS Win!')
        return -1
    print("Draw!")
    return 0







