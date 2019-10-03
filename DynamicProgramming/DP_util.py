from random import randint, random
from time import sleep

def evaluate_policy(policy, n=1000):
    """ Run m episodes with with a given policy, and return the average"""
    data = []
    for i in range(1000):
        data.append(agent(policy))

    print("Average steps to finish: {}".format(sum(data) / len(data)))


def print_board(agent_position):
    fields = list(range(16))
    board = "-----------------\n"
    for i in range(0, 16, 4):
        line = fields[i:i+4]
        for field in line:
            if field == agent_position:
                board += "| A "
            elif field == fields[0] or field == fields[-1]:
                board += "| X "
            else:
                board += "|   "
        board += "|\n"
        board += "-----------------\n"
    print(board)

def create_state_to_state_prime_verbose_map():
    """ For each possible state, determine what the successor state is.

        Return the result as a dictionary.
    """
    l = list(range(16))
    state_to_state_prime = {}
    for i in l:
        if i == 0 or i == 15:
            state_to_state_prime[i] = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
        elif i % 4 == 0:
            state_to_state_prime[i] = {'N': i - 4 if i - 4 in l else i, 'E': i + 1 if i + 1 in l else i, 'S': i + 4 if i + 4 in l else i, 'W': i}
        elif i % 4 == 3:
            state_to_state_prime[i] = {'N': i - 4 if i - 4 in l else i, 'E': i, 'S': i + 4 if i + 4 in l else i, 'W': i - 1 if i - 1 in l else i}
        else:
            state_to_state_prime[i] = {'N': i - 4 if i - 4 in l else i, 'E': i + 1 if i + 1 in l else i, 'S': i + 4 if i + 4 in l else i, 'W': i - 1 if i - 1 in l else i}

    return state_to_state_prime

def create_random_policy():
    """ Policy that returns a random action in all states."""
    return {i: {'N': 0.0, 'E': 0.0, 'S': 0.0, 'W': 0.0} if i == 0 or i == 15 else
        {'N': 0.25, 'E': 0.25, 'S': 0.25, 'W': 0.25} for i in range(16)} # [N, E, S, W]

def create_probability_map():
    """ Return state transition probabilities: p(s', r | s, a)

        I.e. what is the probability of ending up in state s' and recieve reward r,
        given being in state s and performing action a?

        The set is exhaustive that means it contains all possibilities even those not
        allowed by our game. For our simple problem, it contains 1024 values!
    """

    """ Retrun the transition probabilities for all state-action-state pairs.
        Most of these will be zero, as the states aren't adjacent.
        The rest will be 1, as this is a deterministic environment.
    """

    states = list(range(16))
    state_to_state_prime = create_state_to_state_prime_verbose_map()

    probability_map = {}

    for state in states:
        for move in ["N", "E", "S", "W"]:
            for prime in states:
                probability_map[(prime, -1, state, move)] = 0 if prime != state_to_state_prime[state][move] else 1

    return probability_map


def agent(policy, starting_position=None, verbose=False):
    l = list(range(16))
    state_to_state_prime = create_state_to_state_prime_verbose_map()
    agent_position = randint(1, 14) if starting_position is None else starting_position

    step_number = 1
    action_taken = None

    while not (agent_position == 0 or agent_position == 15):

        current_policy = policy[agent_position]
        next_move = random()
        lower_bound = 0
        for action, chance in current_policy.items():
            if next_move < lower_bound + chance:
                if verbose:
                    print("Move: {} Position: {} Action: {}".format(step_number, agent_position, action_taken))
                    print_board(agent_position)
                    print("\n")
                    sleep(0.5)

                agent_position = state_to_state_prime[agent_position][action]
                action_taken = action
                break
            lower_bound = lower_bound + chance

        step_number += 1

    if verbose:
        print("Move: {} Position: {} Action: {}".format(step_number, agent_position, action_taken))
        print_board(agent_position)
        print("Win!")

    return step_number


