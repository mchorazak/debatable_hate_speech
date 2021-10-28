# the file contains one method for each Speech Act.
# Each method chooses randomly one of the pre-defined sentences,
# fills it with data passed through the arguments and returns a string.
import random


def get_cat(number):
    return "hate" if number == 1 else "not hate"


def propose(sent, cat):
    choose = random.randint(0, 2)
    if choose == 0:
        return "I think the sentence %s is %s." % (sent, get_cat(cat))
    elif choose == 1:
        return "%s is %s." % (sent, get_cat(cat))
    elif choose == 2:
        return "The sentence %s is %s." % (sent, get_cat(cat))


def why(sent, cat):
    choose = random.randint(0, 2)
    if choose == 0:
        return "Why is it %s?" % (get_cat(cat))
    elif choose == 1:
        return "Why do you think it is %s?" % (get_cat(cat))
    elif choose == 2:
        return "Why would you say it is %s?" % (get_cat(cat))


def argue_prob(sent, cat, prob):
    choose = random.randint(0, 1)
    if choose == 0:
        return "The model predicted it to be %s with probability %s." % (get_cat(cat), prob)
    elif choose == 1:
        return "My model predicted that %s is %s with probability %s." % (sent, get_cat(cat), prob)


def accept_probability(sent, cat, prob):
    choose = random.randint(0, 1)
    if choose == 0:
        return "I agree it should be %s. I am %s confident." % (get_cat(cat), prob)
    elif choose == 1:
        return "I am %s sure that this is right, %s should be %s. " % (prob, sent, get_cat(cat))


def reject_soft(sent, cat, prob):
    choose = random.randint(0, 1)
    if choose == 0:
        return "I disagree. I think it should be %s." % (get_cat(cat))
    elif choose == 1:
        return "I think you are wrong. I would say it should be %s." % (get_cat(cat))


def reject_probability(sent, cat, prob, orig_prob):
    choose = random.randint(0, 1)
    if choose == 0:
        return "I disagree. I think it should be %s. I am %s confident, what makes me more confident than you." \
               % (get_cat(cat), prob)
    elif choose == 1:
        return "I think you are wrong. I am %s confident it should be %s. " \
               "And my estimate is more certain than yours." % (prob, get_cat(cat))


def skip():
    return "SKIP"


def accept_outcome():
    return "I see, you're right."


def distance(sent, dist, cat):
    choose = random.randint(0, 1)
    if choose == 0:
        return "Well it seems to me that %s should be %s because my model says so and the sum of distances of " \
               "training sentences to this sentence is %s." % (sent, get_cat(cat), dist)
    elif choose == 1:
        return "My model says it should be %s and it can be trusted because the summed distance between the sentence" \
               " and training sentences is %s." % (get_cat(cat), dist)


def accept_distance(dist):
    choose = random.randint(0, 1)
    if choose == 0:
        return "I see, my distance is larger at %s . Perhaps you are right." % dist
    elif choose == 1:
        return "My training data was not so similar to this sentence... The distance is %s." % dist


def reject_distance(dist, cat):
    choose = random.randint(0, 1)
    if choose == 0:
        return "Well, my distance is smaller at %s. That makes my prediction more trustworthy." % dist
    elif choose == 1:
        return "But my training data was more similar to this sentence, the distance is %s so it should be %s." \
               % (dist, get_cat(cat))
