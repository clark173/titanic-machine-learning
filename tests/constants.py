# Sample feature sets
TEST_FEATURE_SETS = {
    'Fare': [7.25, 71.2833, 7.925],
    'Embarked': [1, 2],
    'Age': [22, 38, 26],
    'Parch': [0],
    'Pclass': [1, 2, 3],
    'Sex': [0, 1],
    'SibSp': [1, 0],
}
TRAIN_FEATURE_SETS = {
    'Fare': [7.25, 71.2833, 7.925],
    'Embarked': [1, 2],
    'Age': [22, 38, 26],
    'Parch': [0],
    'Pclass': [1, 2, 3],
    'Sex': [0, 1],
    'Survived': [0, 1],
    'SibSp': [1, 0],
}

# Passenger headers
TRAIN_HEADER_LIST = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TRAIN_LIST_FOR_PASS_ID = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TEST_HEADER_LIST = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TEST_LIST_FOR_PASS_ID = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Passenger ID Lists
TRAIN_PASSENGER_ID_LIST = [1, 2, 3]
TEST_PASSENGER_ID_LIST = [1, 2, 3]

# Data field filters
GENDER_PARSE = {'male': 0, 'female': 1}
PORT_OF_EMBARKMENT = {'Q': 0, 'S': 1, 'C': 2}

# Feature headers
COLUMN_X_DATA_SET = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
COLUMN_Y_DATA_SET = ['Survived']

# Input data files
SAMPLE_DATA = 'tests/sample_training.csv'
