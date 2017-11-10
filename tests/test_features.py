import constants as c
from features import Features


class TestFeatures:
    def test_getting_training_passengers(self):
        passengers = Features()._get_passengers(c.SAMPLE_DATA, predict=False)

        assert passengers.header == c.TRAIN_HEADER_LIST
