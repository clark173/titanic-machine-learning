import constants as c
import csv
import pandas


class Passengers:
    def __init__(self, input_file):
        self._header = c.HEADER
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
        for gender in c.GENDER_PARSE.keys():
            data = data.replace(gender, c.GENDER_PARSE[gender])
        for port in c.PORT_OF_EMBARKMENT.keys():
            data = data.replace(port, c.PORT_OF_EMBARKMENT[port])
        data = data.where((pandas.notnull(data)), None)
        return data

    def _build_feature_sets(self, data):
        for header in self._header:
            self._feature_sets[header] = list(set(data[header].tolist()))
