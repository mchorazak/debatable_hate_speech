from enum import Enum


class Move:
    def __init__(self, who=None, act=None, argument=None):
        self.argument = argument
        self.WHO = who
        self.ACT = act


class Act(Enum):
    PROPOSE = 0
    WHY = 1
    ARGUE_PROB = 2
    ACCEPT_PROB = 3
    SOFT_REJECT_PROB = 4
    REJECT_PROB = 5
    ARGUE_DIST = 6
    ACCEPT_DIST = 7
    REJECT_DIST = 8
    ACCEPT_OUTCOME = 9
    SKIP = 10


class WHO(Enum):
    Model1 = 1
    Model2 = 2


class CAT(Enum):
    NO_HATE = 0
    HATE = 1


class KnowledgeBase:
    my_prob = -1
    my_dist = -1
    their_prob = -1
    their_dist = -1
    my_class = -1
