import numpy as np


opposites = {
    "North" : "South",
    "South": "North",
    "West": "East",
    "East": "West",
}


class Sequence:
    """ Continues until one failure is found."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        raise NotImplementedError


class Selector:
    """ Continues until one success is found."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        raise NotImplementedError


class CheckValid:
    """ Check whether <direction> is a valid action for PacMan
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        raise NotImplementedError


class CheckDanger:
    """ Check whether there is a ghost in <direction>, or any of the adjacent fields.
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        raise NotImplementedError


class ActionGo:
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """
    def __init__(self, direction="Random"):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        raise NotImplementedError


class ActionGoNot:
    """ Go in a random direction that isn't <direction>
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        raise NotImplementedError


class DecoratorInvert:
    def __call__(self, arg):
        return not arg





def parse_node(genome, parent=None):
    if isinstance(genome[0], list):
        parse_node(genome[0], parent)
        parse_node(genome[1:], parent)

    elif genome[0] is "SEQ":
        if parent is not None:
            node = parent.add_child(Sequence(parent))
        else:
            node = Sequence(parent)
            parent = node
        parse_node(genome[1:], node)

    elif genome[0] is "SEL":
        if parent is not None:
            node = parent.add_child(Selector(parent))
        else:
            node = Selector(parent)
            parent = node
        parse_node(genome[1:], node)

    elif genome[0].startswith("Valid"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckValid(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("Danger"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckDanger(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("GoNot"):
        arg = genome[0].split('.')[-1]
        parent.add_child(ActionGoNot(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("Go"):
        arg = genome[0].split('.')[-1]
        parent.add_child(ActionGo(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)


    elif genome[0] is ("Invert"):
        arg = genome[0].split('.')[-1]
        parent.add_child(DecoratorInvert(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    else:
        print("Unrecognized in ")
        raise Exception

    return parent




