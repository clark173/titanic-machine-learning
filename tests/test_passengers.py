import constants as c
from collections import Counter
from passengers import Passengers


class TestPassengers:
    def test_passengers_integration_training_set(self):
        passengers = Passengers(c.SAMPLE_DATA, predict=False)

        assert passengers.header == c.TRAIN_HEADER_LIST
        for feature, value in passengers.feature_sets.items():
            assert Counter(c.TRAIN_FEATURE_SETS[feature]) == Counter(value)
        assert len(passengers.feature_sets) == len(c.TRAIN_FEATURE_SETS)
        assert passengers.passenger_id_list == c.TRAIN_PASSENGER_ID_LIST
        assert not passengers.data_set.empty

    def test_passengers_integration_testing_set(self):
        passengers = Passengers(c.SAMPLE_DATA, predict=True)

        assert passengers.header == c.TEST_HEADER_LIST
        for feature, value in passengers.feature_sets.items():
            assert Counter(c.TEST_FEATURE_SETS[feature]) == Counter(value)
        assert len(passengers.feature_sets) == len(c.TEST_FEATURE_SETS)
        assert passengers.passenger_id_list == c.TEST_PASSENGER_ID_LIST
        assert not passengers.data_set.empty
