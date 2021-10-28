# Implements the rules of argumentation for a debater.
# 'top_discuss()' is called from outside and is given a previous move to work with.
# The method redirects to other methods based on what kind of move it is.
# Next stage is one of the "answer_to_X()" methods, where X is a speech act name.
# They either calculate the model's response or call a more specific function to do that.
# In any case, an instance of Move(a response) is returned
# up the calling stack to the method that first  called "top_discuss()"

import json
from math import sqrt
from language_generator import print_argument
import random
from helpers.protocol import Move, Act, CAT, KnowledgeBase
from scipy import spatial
import helpers.config as cf


class DebaterBase:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.probabilities = []
        self.predictions = []
        self.random_int = random.randint
        self.distance_used = False
        self.probability_used = False
        self.claim = None
        self.knowledge_base = KnowledgeBase()
        self.first_why = True

    def run(self):
        print("running base")

    # flags restricting use of argument to just once per discussion.
    def reset_flags(self):
        self.first_why = True
        self.distance_used = False
        self.probability_used = False

    # Show plots of accuracy and loss according to "history"
    @staticmethod
    def show_history(history):
        import matplotlib.pyplot as plt
        # accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def euclidean_distance(self, x, y):
        return 0.001 * sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

    def calculate_distance(self, sentence_id):
        sum = 0
        for index, row in self.data.x_train.iterrows():
            sum += spatial.distance.cosine(self.data.x_test.iloc[sentence_id], row)
            # sum += self.euclidean_distance(self.data.x_test.iloc[sentence_id], row)
        return sum

    def get_similar_sentence(self, sentence):
        min_dist = self.euclidean_distance(sentence, self.data.x_train.iloc[0])
        similar = self.data.train_sentences.iloc[0]
        for index, row in self.data.x_train.iterrows():
            if spatial.distance.cosine(sentence, row) < min_dist:
                similar = self.data.train_sentences[index]
        return similar

    def initiate_discussion(self, sentence_id):
        move = Move(who=self.name, act=Act.PROPOSE)
        if self.probabilities[sentence_id] >= 0.5:
            argument = {
                "sent": self.data.test_sentences[sentence_id],
                "cat": CAT.HATE.value,
                "sentence_id": sentence_id
            }
            self.knowledge_base.my_class = CAT.HATE
        else:
            argument = {
                "sent": self.data.test_sentences[sentence_id],
                "cat": CAT.NO_HATE.value,
                "sentence_id": sentence_id
            }
            self.knowledge_base.my_class = CAT.NO_HATE
        move.argument = json.dumps(argument)
        print_argument(move)
        return move

    def argue_with_probability(self, argument):
        self.probability_used = True
        json_arg = json.loads(argument.argument)
        prob = -1
        move = Move(who=self.name, act=Act.ARGUE_PROB)
        # GET THE PROBABILITY FOR THE SENTENCE TO BE DISCUSSED
        for x in range(len(self.data.test_sentences)):
            if json_arg["sent"] == self.data.test_sentences[x]:
                if self.probabilities[x] >= 0.5:
                    prob = self.probabilities[x]
                if self.probabilities[x] < 0.5:
                    prob = 1 - self.probabilities[x]
                break
        json_arg["prob"] = str(prob)
        move.argument = json.dumps(json_arg)
        self.knowledge_base.my_prob = prob
        print_argument(move)

        return move

    def answer_to_probability(self, argument):
        json_arg = json.loads(argument.argument)
        self.knowledge_base.their_prob = json_arg["prob"]
        prob = -1
        move = Move(who=self.name)
        # GET THE PROBABILITY FOR THE SENTENCE TO BE DISCUSSED
        for x in range(len(self.data.test_sentences)):
            if json_arg["sent"] == self.data.test_sentences[x]:
                prob = self.probabilities[x]
                break
        # FOUR CASES OF AGREEMENT
        response = {
            "sent": json_arg["sent"],
            "original_prob": json_arg["prob"],
            "sentence_id": json_arg["sentence_id"]
        }
        if json_arg["cat"] == CAT.HATE.value and prob >= 0.5:  # agree that hate
            move.ACT = Act.ACCEPT_PROB
            self.knowledge_base.my_class = CAT.HATE
            response["cat"] = CAT.HATE.value
            response["prob"] = str(prob)
        elif json_arg["cat"] == CAT.NO_HATE.value and prob < 0.5:  # agree that not hate
            move.ACT = Act.ACCEPT_PROB
            self.knowledge_base.my_class = CAT.NO_HATE
            response["cat"] = CAT.NO_HATE.value
            response["prob"] = str(1 - prob)
        elif json_arg["cat"] == CAT.HATE.value and prob < 0.5:  # not agree that hate
            if cf.OPTIMAL_DISCUSSION:
                if (1 - prob) > float(json_arg["prob"]):
                    # reject and support
                    move.ACT = Act.REJECT_PROB
                    self.knowledge_base.my_class = CAT.NO_HATE
                    response["cat"] = CAT.NO_HATE.value
                    response["prob"] = str(1 - prob)
                else:
                    # accept defeat
                    move.ACT = Act.ACCEPT_OUTCOME
                    self.knowledge_base.my_class = CAT.HATE
                    response["cat"] = CAT.HATE.value
                    response["prob"] = str(1-prob)
            else:
                if (1 - prob) > float(json_arg["prob"]):
                    # reject and support
                    move.ACT = Act.REJECT_PROB
                else:
                    # reject without justification
                    move.ACT = Act.SOFT_REJECT_PROB
                self.knowledge_base.my_class = CAT.NO_HATE
                response["cat"] = CAT.NO_HATE.value
                response["prob"] = str(1 - prob)
        elif json_arg["cat"] == CAT.NO_HATE.value and prob >= 0.5:  # not agree that not hate
            if cf.OPTIMAL_DISCUSSION:
                if prob > float(json_arg["prob"]):
                    # reject and support
                    move.ACT = Act.REJECT_PROB
                    self.knowledge_base.my_class = CAT.HATE
                    response["cat"] = CAT.HATE.value
                    response["prob"] = str(prob)
                else:
                    # accept defeat
                    move.ACT = Act.ACCEPT_OUTCOME
                    self.knowledge_base.my_class = CAT.NO_HATE
                    response["cat"] = CAT.NO_HATE.value
                    response["prob"] = str(prob)
            else:
                if prob > float(json_arg["prob"]):
                    # reject and support
                    move.ACT = Act.REJECT_PROB
                else:
                    # reject without justification
                    move.ACT = Act.SOFT_REJECT_PROB
                self.knowledge_base.my_class = CAT.HATE
                response["cat"] = CAT.HATE.value
                response["prob"] = str(prob)

        self.knowledge_base.my_class = response["cat"]
        self.knowledge_base.my_prob = response["prob"]
        move.argument = json.dumps(response)
        print_argument(move)
        return move

    def ask_why(self, argument):
        move = Move(who=self.name)
        move.ACT = Act.WHY
        move.argument = argument.argument
        print_argument(move)
        return move

    def answer_to_why(self, argument):
        if self.first_why:
            self.first_why = False
            return self.argue_with_probability(argument)
        else:
            return self.get_distance_argument(argument)

    def answer_to_reject_prob(self, argument):
        move = Move(who=self.name, act=Act.ACCEPT_OUTCOME, argument=argument.argument)
        print_argument(move)
        return move

    def get_distance_argument(self, argument):
        self.distance_used = True
        json_arg = json.loads(argument.argument)
        move = Move(who=self.name, act=Act.ARGUE_DIST)
        distance = self.calculate_distance(json_arg["sentence_id"])
        response = {
            "sent": json_arg["sent"],
            "sentence_id": json_arg["sentence_id"],
            "dist": str(distance)
        }
        if self.knowledge_base.my_class == CAT.HATE:
            response["cat"] = CAT.HATE.value
        else:
            response["cat"] = CAT.NO_HATE.value
        self.knowledge_base.my_dist = distance
        move.argument = json.dumps(response)
        print_argument(move)
        return move

    def answer_to_soft_reject_prob(self, argument):
        while True:
            n = 0 if cf.OPTIMAL_DISCUSSION else random.randint(0, 1)
            if n == 0:
                return self.ask_why(argument)
            if n == 1 and not self.distance_used:
                return self.get_distance_argument(argument)

    def answer_to_accept(self, argument):
        move = Move(who=self.name, act=Act.SKIP, argument=argument.argument)
        print_argument(move)
        return move

    def answer_to_skip(self, argument, my_previous_argument):
        if my_previous_argument.ACT == Act.ACCEPT_OUTCOME or my_previous_argument.ACT == Act.ACCEPT_PROB\
                or my_previous_argument.ACT == Act.ACCEPT_DIST:
            move = Move(who=self.name, act=Act.SKIP, argument=argument.argument)
            print_argument(move)
            return move

    def answer_to_distance(self, argument):
        self.distance_used = True
        json_arg = json.loads(argument.argument)
        move = Move(who=self.name, act=Act.SKIP)

        distance = self.calculate_distance(json_arg["sentence_id"])

        response = {
            "sent": json_arg["sent"],
            "sentence_id": json_arg["sentence_id"],
            "dist": str(distance),
            "previous": argument.argument
        }
        self.knowledge_base.their_distance = float(json_arg["dist"])
        if distance <= float(json_arg["dist"]):
            move.ACT = Act.REJECT_DIST
            response["cat"] = CAT.HATE.value if json_arg["cat"] == CAT.NO_HATE.value else CAT.NO_HATE.value
        else:
            move.ACT = Act.ACCEPT_DIST
            response["cat"] = json_arg["cat"]
        move.argument = json.dumps(response)
        print_argument(move)
        return move

    def answer_to_rejected_distance(self, argument):
        json_arg = json.loads(argument.argument)
        self.knowledge_base.their_dist = json_arg["dist"]
        response = {
            "sent": json_arg["sent"],
            "sentence_id": json_arg["sentence_id"],
            "previous": argument.argument,
            "cat": json_arg["cat"]
        }
        move = Move(who=self.name, act=Act.ACCEPT_OUTCOME)
        move.argument = json.dumps(response)
        print_argument(move)
        return move

    # Top level method of a move. Initiates the discusion if no moves present in the list.
    # Otherwise, depending on the ACT of the previous argument,
    # direct the logic to respective  handling method.
    def discuss_top(self, move_list, sentence_id):
        if not move_list:
            # initiate discussion with a proposal
            return self.initiate_discussion(sentence_id)
        else:
            # get answer to the last argument
            argument = move_list[len(move_list) - 1]
            if argument.ACT == Act.PROPOSE:
                return self.ask_why(argument)
            elif argument.ACT == Act.ARGUE_PROB:
                return self.answer_to_probability(argument)
            elif argument.ACT == Act.WHY:
                return self.answer_to_why(argument)
            elif argument.ACT == Act.SOFT_REJECT_PROB:
                return self.answer_to_soft_reject_prob(argument)
            elif argument.ACT == Act.REJECT_PROB:
                return self.answer_to_reject_prob(argument)
            elif argument.ACT == Act.ACCEPT_OUTCOME or argument.ACT == Act.ACCEPT_PROB or argument.ACT == Act.ACCEPT_DIST:
                return self.answer_to_accept(argument)
            elif argument.ACT == Act.SKIP:
                return self.answer_to_skip(argument, move_list[len(move_list) - 2])
            elif argument.ACT == Act.ARGUE_DIST:
                return self.answer_to_distance(argument)
            elif argument.ACT == Act.REJECT_DIST:
                return self.answer_to_rejected_distance(argument)
