from collections import Counter
from passengers import Passengers


FEATURE_SETS = {
    'Fare': [7.25, 71.2833, 7.925],
    'Name': [
             'Braund, Mr. Owen Harris',
             'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
             'Heikkinen, Miss. Laina'
            ],
    'Embarked': [1, 2],
    'Age': [22, 38, 26],
    'Parch': [0],
    'Pclass': [1, 2, 3],
    'Sex': [0, 1],
    'Survived': [0, 1],
    'SibSp': [1, 0],
    'PassengerId': [1, 2, 3],
    'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
    'Cabin': [None, 'C85']
}
HEADERS = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
SAMPLE_DATA = 'tests/sample_training.csv'


class TestPassengers:
    def test_passengers_integration(self):
        passengers = Passengers(SAMPLE_DATA)

        assert passengers.header == HEADERS
        for feature, value in passengers.feature_sets.items():
            assert Counter(FEATURE_SETS[feature]) == Counter(value)
        assert len(passengers.feature_sets) == len(FEATURE_SETS)
