import constants as c
from collections import Counter
from passengers import Passengers


class TestPassengers:
    def test_passengers_integration(self):
        passengers = Passengers(c.SAMPLE_DATA)

        assert passengers.header == c.HEADERS
        for feature, value in passengers.feature_sets.items():
            assert Counter(c.FEATURE_SETS[feature]) == Counter(value)
        assert len(passengers.feature_sets) == len(c.FEATURE_SETS)
