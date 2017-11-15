# Passenger headers
TRAIN_HEADER_LIST = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TRAIN_LIST_FOR_PASS_ID = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TEST_HEADER_LIST = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TEST_LIST_FOR_PASS_ID = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Passenger headers without age
TRAIN_HEADER_NO_AGE = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
TRAIN_LIST_PASS_ID_NO_AGE = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
TEST_HEADER_NO_AGE = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
TEST_LIST_PASS_ID_NO_AGE = ['PassengerId', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Data field filters
GENDER_PARSE = {'male': 0, 'female': 1}
PORT_OF_EMBARKMENT = {'Q': 0, 'S': 1, 'C': 2}

# Feature headers
COLUMN_X_DATA_SET = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
COLUMN_X_DATA_SET_NO_AGE = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
COLUMN_Y_DATA_SET = ['Survived']

# Input data files
TEST_FILE = 'test.csv'
TRAIN_FILE = 'train.csv'

# Survival outcome
DIED = 0
SURVIVED = 1
