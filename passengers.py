import constants as c
import pandas


class Passengers:
    def __init__(self, input_file, predict):
        self._passenger_id_list = []
        self._data_set = None
        if predict:
            self._header = c.TEST_HEADER_LIST
            self._pass_id_list = c.TEST_LIST_FOR_PASS_ID
        else:
            self._header = c.TRAIN_HEADER_LIST
            self._pass_id_list = c.TRAIN_LIST_FOR_PASS_ID
        self._feature_sets = {}
        passenger_data = pandas.read_csv(input_file, header=0)
        self._data_set = self._edit_passenger_data(passenger_data)
        self._build_feature_sets(self._data_set)

    @property
    def passenger_id_list(self):
        return self._passenger_id_list

    @property
    def data_set(self):
        return self._data_set

    @property
    def header(self):
        return self._header

    @property
    def feature_sets(self):
        return self._feature_sets

    def _edit_passenger_data(self, data):
        temp_passenger_id_data = data[self._pass_id_list].dropna()
        self._passenger_id_list = temp_passenger_id_data['PassengerId'].tolist()
        data = data[self._header].dropna()
        for gender in c.GENDER_PARSE.keys():
            data = data.replace(gender, c.GENDER_PARSE[gender])
        for port in c.PORT_OF_EMBARKMENT.keys():
            data = data.replace(port, c.PORT_OF_EMBARKMENT[port])
        return data

    def _build_feature_sets(self, data):
        for header in self._header:
            self._feature_sets[header] = list(set(data[header].tolist()))
