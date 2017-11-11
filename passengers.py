import csv
import pandas


HEADER = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
          'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
GENDER_PARSE = {'male': 0, 'female': 1}
PORT_OF_EMBARKMENT = {'Q': 0, 'S': 1, 'C': 2, '': -1}


class Passengers:
    def __init__(self, input_file):
        self._header = HEADER
        self._feature_sets = {}

        passenger_data = pandas.read_csv(input_file, header=0)
        passenger_data = self._edit_passenger_data(passenger_data)
        self._build_feature_sets(passenger_data)

    @property
    def header(self):
        return self._header

    @property
    def feature_sets(self):
        return self._feature_sets

    def _edit_passenger_data(self, data):
        for gender in GENDER_PARSE.keys():
            data = data.replace(gender, GENDER_PARSE[gender])
        for port in PORT_OF_EMBARKMENT.keys():
            data = data.replace(port, PORT_OF_EMBARKMENT[port])
        data = data.where((pandas.notnull(data)), None)
        return data

    def _build_feature_sets(self, data):
        for header in self._header:
            self._feature_sets[header] = list(set(data[header].tolist()))
