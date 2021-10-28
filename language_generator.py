import json
from helpers.protocol import Act
import sentence_base
from helpers import config


# based on the speech ACT of the move, get a string from the respective method
def choose_argument(move):
    json_arg = json.loads(move.argument)
    if move.ACT == Act.PROPOSE:
        return sentence_base.propose(json_arg["sent"], json_arg["cat"])
    elif move.ACT == Act.WHY:
        return sentence_base.why(json_arg["sent"], json_arg["cat"])
    elif move.ACT == Act.ARGUE_PROB:
        return sentence_base.argue_prob(json_arg["sent"], json_arg["cat"], json_arg["prob"])
    elif move.ACT == Act.ACCEPT_PROB:
        return sentence_base.accept_probability(json_arg["sent"], json_arg["cat"], json_arg["prob"])
    elif move.ACT == Act.SOFT_REJECT_PROB:
        return sentence_base.reject_soft(json_arg["sent"], json_arg["cat"], json_arg["prob"])
    elif move.ACT == Act.REJECT_PROB:
        return sentence_base.reject_probability(json_arg["sent"], json_arg["cat"], json_arg["prob"],
                                                json_arg["original_prob"])
    elif move.ACT == Act.SKIP:
        return "SKIP"
    elif move.ACT == Act.ACCEPT_OUTCOME:
        return sentence_base.accept_outcome()
    elif move.ACT == Act.ARGUE_DIST:
        return sentence_base.distance(json_arg["sent"], json_arg["dist"], json_arg["cat"])
    elif move.ACT == Act.ACCEPT_DIST:
        return sentence_base.accept_distance(json_arg["dist"])
    elif move.ACT == Act.REJECT_DIST:
        return sentence_base.reject_distance(json_arg["dist"], json_arg["cat"])


# Print the move using natural language if not restricted by the verbosity flag "--discuss".
# Add the name of the model before printing the sentence.
def print_argument(move):
    if config.PRINT_CONVERSATION:  # verbosity flag
        print(move.WHO, ": ", choose_argument(move))
