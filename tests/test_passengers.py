from passengers import Passenger, Passengers


FEATURE_SETS = {
    'Fare': list(set([7.25, 71.2833, 7.925])),
    'Name': list(set([
             'Braund, Mr. Owen Harris',
             'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
             'Heikkinen, Miss. Laina'
            ])),
    'Embarked': list(set(['S', 'C'])),
    'Age': list(set([22, 38, 26])),
    'Parch': list(set([0])),
    'Pclass': list(set([1, 2, 3])),
    'Sex': list(set(['male', 'female'])),
    'Survived': list(set([0, 1])),
    'SibSp': list(set([1, 0])),
    'PassengerId': list(set([1, 2, 3])),
    'Ticket': list(set(['A/5 21171', 'PC 17599', 'STON/O2. 3101282'])),
    'Cabin': list(set([None, 'C85']))
}
HEADERS = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
NUM_PASSENGERS = 3
SAMPLE_DATA = 'tests/sample_training.csv'


class TestPassenger:
    def test_passenger_creation(self):
        passengerid = 810
        survived = 1
        pclass = 2
        name = "John Doe"
        sex = "male"
        age = 28
        sibsp = 0
        parch = 0
        ticket = "148625"
        fare = 8.05
        cabin = ""
        embarked = "S"

        passenger = Passenger(passengerid, survived, pclass, name, sex, age,
                              sibsp, parch, ticket, fare, cabin, embarked)

        assert passenger.passengerid == passengerid
        assert passenger.survived == survived
        assert passenger.pclass == pclass
        assert passenger.name == name
        assert passenger.sex == sex
        assert passenger.age == age
        assert passenger.sibsp == sibsp
        assert passenger.parch == parch
        assert passenger.ticket == ticket
        assert passenger.fare == fare
        assert passenger.cabin == cabin
        assert passenger.embarked == embarked


class TestPassengers:
    def test_passengers_integration(self):
        passengers = Passengers(SAMPLE_DATA)

        assert passengers.header == HEADERS
        assert passengers.feature_sets == FEATURE_SETS

        num_passengers = 0
        for passenger in passengers:
            num_passengers += 1
        assert num_passengers == NUM_PASSENGERS
