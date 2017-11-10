import csv
import pandas


class Passenger:
    def __init__(self, passengerid, survived, pclass, name, sex, age, sibsp,
                 parch, ticket, fare, cabin, embarked):
        self._passengerid = passengerid
        self._survived = survived
        self._pclass = pclass
        self._name = name
        self._sex = sex
        self._age = age
        self._sibsp = sibsp
        self._parch = parch
        self._ticket = ticket
        self._fare = fare
        self._cabin = cabin
        self._embarked = embarked

    @property
    def passengerid(self):
        return self._passengerid

    @property
    def survived(self):
        return self._survived

    @property
    def pclass(self):
        return self._pclass

    @property
    def name(self):
        return self._name

    @property
    def sex(self):
        return self._sex

    @property
    def age(self):
        return self._age

    @property
    def sibsp(self):
        return self._sibsp

    @property
    def parch(self):
        return self._parch

    @property
    def ticket(self):
        return self._ticket

    @property
    def fare(self):
        return self._fare

    @property
    def cabin(self):
        return self._cabin

    @property
    def embarked(self):
        return self._embarked


class Passengers:
    def __init__(self, input_file):
        self._passengers = []
        self._header = []
        self._feature_sets = {}

        passenger_data = pandas.read_csv(input_file, header=0)
        passenger_data = passenger_data.where((pandas.notnull(passenger_data)),
                                              None)
        self._read_input_file(input_file)
        self._build_feature_sets(passenger_data)

    def __expr__(self):
        return self._passengers

    def __iter__(self):
        return iter(self._passengers)

    @property
    def header(self):
        return self._header

    @property
    def feature_sets(self):
        return self._feature_sets

    def _read_input_file(self, input_file):
        with open(input_file, 'rb') as passenger_list:
            reader = csv.reader(passenger_list)
            self._header = next(reader, None)  # Save the header line
            for passenger in reader:
                self._passengers.append(Passenger(*passenger))

    def _build_feature_sets(self, data):
        for header in self._header:
            self._feature_sets[header] = list(set(data[header].tolist()))
