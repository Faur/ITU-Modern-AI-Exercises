import DP_util

""" Evaluate a random policy."""
random_policy = DP_util.create_random_policy()
DP_util.evaluate_policy(random_policy)

""" Uncomment to visualize a run."""
DP_util.agent(DP_util.create_random_policy(), verbose=True)


def policy_iteration(theta=0.01, discount_rate=0.5):
    """"""
    # transition probabilities: p(s', r | s, a)
    # Implemented as a dictionary, with key formatted as:
    #   [next_state, reward, state, acton]
    # Note that the reward is always -1
    #
    # EG. state_transition_probabilities[2, -1, 1, 'E'] == 1, as
    # standing in state 1, and going east (action 'E') will move you to
    # state 2.
    state_transition_probabilities = DP_util.create_probability_map()

    # State transitions - i.e.
    # EG. s_to_sprime[1]['E'] what is the next state, if the agent
    # is in staet 1, and moves east (performs action 'E').
    s_to_sprime = DP_util.create_state_to_state_prime_verbose_map()

    """ #1: Initialization 

        V_s is a dictionary with the that contains the value of each state.
        We will use it to create better policy which will choose 
        next state based on maximum value. Remember the terminal 
        states must always have a value of 0.

        policy is a dictionary of the action probabilities for each state.
    """
    V_s = {i: 0 for i in range(16)}  # Everything zero
    policy = DP_util.create_random_policy()  # Random actions

    done = False
    while not done:
        """ # 2: Policy Evaluation.

            Updates the value function V_s, until the change is smaller than theta.
        """

        """ YOUR CODE HERE! """


        """ #3: Policy improvement

            Update the policy if necessary. If the policy is stable (doesn't change)
            set done to True.          
        """
        policy_stable = True

        """ YOUR CODE HERE! """

        if policy_stable:
            done = True

    return V_s, policy


V_s, policy = policy_iteration()

DP_util.evaluate_policy(policy)
DP_util.agent(policy, verbose=True)

