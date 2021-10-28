# The tests are examples of different situations that can occur.
# Each case has declared probabilities and config flags are set for predictable outcome.

import unittest
from helpers.data_loader import get_data, introduce_variation
from model_lstm import LSTMDebater
from model_bdir import BIDIRDebater
from helpers.protocol import WHO
from main import debate
from helpers import config


class DebateTests(unittest.TestCase):

    def setUp(self):
        import time
        config.OPTIMAL_DISCUSSION = True
        config.PRINT_CONVERSATION = True
        config.PRINT_OUTCOME = True
        config.PLOTS = False
        time.sleep(1)
        self.sentence_id = 0
        self.data_original = get_data()
        self.data_modified = introduce_variation(get_data())

    def test_quick_agree_on_probability(self):
        print("\nQuick agree:")
        lstm = LSTMDebater(WHO.Model1, self.data_original)
        bdir = BIDIRDebater(WHO.Model2, self.data_modified)
        lstm.probabilities = [0.7]
        bdir.probabilities = [0.7]
        debate(lstm, bdir, self.sentence_id)

    def test_convince_with_distance(self):
        print("\nOne model has better probability and better distance:")
        config.OPTIMAL_DISCUSSION = False
        lstm = LSTMDebater(WHO.Model1, self.data_modified)
        bdir = BIDIRDebater(WHO.Model2, self.data_original)
        lstm.probabilities = [0.3]
        bdir.probabilities = [0.6]
        debate(lstm, bdir, self.sentence_id)

    def test_soft_reject_probability(self):
        config.OPTIMAL_DISCUSSION = False
        print("\nThe first model has better accuracy but worse distance:")
        lstm = LSTMDebater(WHO.Model1, self.data_original)
        bdir = BIDIRDebater(WHO.Model2, self.data_modified)
        lstm.probabilities = [0.8]
        bdir.probabilities = [0.3]
        debate(lstm, bdir, self.sentence_id)

    def test_disagree_with_probability_optimal(self):
        config.OPTIMAL_DISCUSSION = False
        print("\nThe first model has better accuracy but worse distance:")
        lstm = LSTMDebater(WHO.Model1, self.data_original)
        bdir = BIDIRDebater(WHO.Model2, self.data_modified)
        lstm.probabilities = [0.8]
        bdir.probabilities = [0.1]
        debate(lstm, bdir, self.sentence_id)


if __name__ == '__main__':
    unittest.main(verbosity=1)
