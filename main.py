import csv
import json
from model_lstm import LSTMDebater
from model_bdir import BIDIRDebater
from helpers.protocol import WHO, Act
from helpers import printing, config
import random


def are_last_2_skip(move_list):
    if not move_list:
        return False
    else:
        last_2_skip = True
        if move_list[(len(move_list) - 1)].ACT != Act.SKIP:
            last_2_skip = False
        if move_list[(len(move_list) - 2)].ACT != Act.SKIP:
            last_2_skip = False
        return last_2_skip


# The debate function that lets MODEL1 and MODEL2 discuss the test sentence of id SENTENCE_ID.
# Models take turns to make arguments which are stored in the MOVE_LIST
def debate(model1, model2, sentence_id):
    model1.reset_flags()
    model2.reset_flags()
    move_list = []                      # initiate the list of moves
    if config.OPTIMAL_DISCUSSION:
        turn = WHO.Model1
    else:
        turn = random.choice(list(WHO))     # randomly choose who starts the discussion
    while not are_last_2_skip(move_list):   # the loop asking the models in turns to make an argument
        if turn == WHO.Model1:
            move_list.append(model1.discuss_top(move_list, sentence_id))
            turn = WHO.Model2
        elif turn == WHO.Model2:
            move_list.append(model2.discuss_top(move_list, sentence_id))
            turn = WHO.Model1
    json_arg = json.loads(move_list[len(move_list)-1].argument)  # load data from the last argument to print the outcome
    if config.PRINT_OUTCOME:    # print final judgement on the sentence
        print("OUTCOME: ", json_arg["sent"], " is ", ("hate.\n" if json_arg["cat"] == 1 else "not hate.\n"))
    return json_arg["cat"] == 1


# Main method of the program. Creates models, runs them and initiates the discussion.
# Finally, analysing methods are called.
def main():
    from helpers.data_loader import get_data, introduce_variation
    # construct two models
    data_1 = get_data()
    lstm = LSTMDebater(WHO.Model1, data_1)
    data_2 = get_data()
    data_2 = introduce_variation(data_2)    # DATA_2 has shuffled train and val sets. Test set remains the same.
    bdir = BIDIRDebater(WHO.Model2, data_2)

    lstm.run()
    bdir.run()

    printing.print_scores(lstm)
    printing.print_scores(bdir)

    printing.disputed(lstm, bdir)

    length = len(data_1.y_test)
    print("Discussion started...")
    discussion_results_optimal = []
    counter = 0
    for sentence_id in range(length):
        discussion_results_optimal.append(int(debate(lstm, bdir, sentence_id)))
        print(counter)
        counter += 1
    print("...discussion finished.")
    printing.print_scores1(data_1.y_test[0:length], discussion_results_optimal)

    print("Discussion started...")
    discussion_results_sub = []
    config.OPTIMAL_DISCUSSION = False
    counter = 0
    for sentence_id in range(length):
        discussion_results_sub.append(int(debate(lstm, bdir, sentence_id)))
        print(counter)
        counter += 1
    print("...discussion finished.")
    printing.print_scores1(data_1.y_test[0:length], discussion_results_sub)

    with open('test_classifications.csv', 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(lstm.predictions)
        writer.writerow(bdir.predictions)
        writer.writerow(discussion_results_optimal)
        writer.writerow(discussion_results_sub)

    config.sound_signal()  # play sound signal for notification


# Entry point of the program.
if __name__ == '__main__':
    main()
