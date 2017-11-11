# Passenger headers
HEADER = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
          'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# Data field filters
GENDER_PARSE = {'male': 0, 'female': 1}
PORT_OF_EMBARKMENT = {'Q': 0, 'S': 1, 'C': 2}

# Feature headers
COLUMN_X_DATA_SET = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
COLUMN_Y_DATA_SET = ['Survived']

# Input data files
TEST_FILE = 'test.csv'
TRAIN_FILE = 'train.csv'

# Survival outcome
DIED = 0
SURVIVED = 1
