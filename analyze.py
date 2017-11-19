import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score


TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

TITLE_DICTIONARY = {
    'Capt': 'Official',
    'Col': 'Official',
    'Major': 'Official',
    'Jonkheer': 'Royalty',
    'Don': 'Royalty',
    'Sir': 'Royalty',
    'Dr': 'Serveant',
    'Rev': 'Serveant',
    'the Countess': 'Royalty',
    'Dona': 'Royalty',
    'Mme': 'Misses',
    'Mlle': 'Miss',
    'Ms': 'Misses',
    'Mr': 'Mister',
    'Mrs': 'Misses',
    'Miss': 'Miss',
    'Master': 'Master',
    'Lady': 'Royalty'
}

def read_csv_to_pandas_dataframe(filename):
    return pd.read_csv(filename)

def get_targets(data):
    return data.Survived

def combine_dataframes(train_data, test_data):
    train_data.drop('Survived', 1, inplace=True)
    combined = train_data.append(test_data)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    return combined

def extract_titles(data):
    data['Title'] = data['Name'].map(lambda name: \
        name.split(',')[1].split('.')[0].strip())
    data['Title'] = data.Title.map(TITLE_DICTIONARY)
    return data

def replace_age(row, grouped_median):
    sex = row['Sex']
    pclass = row['Pclass']
    title = row['Title']

    ### Females
    # Pclass of 1
    if sex == 'female' and pclass == 1 and title == 'Miss':
        return grouped_median.loc['female', 1, 'Miss']['Age']
    if sex == 'female' and pclass == 1 and title == 'Misses':
        return grouped_median.loc['female', 1, 'Misses']['Age']
    if sex == 'female' and pclass == 1 and title == 'Misses':
        return grouped_median.loc['female', 1, 'Royalty']['Age']
    # Pclass of 2
    if sex == 'female' and pclass == 2 and title == 'Miss':
        return grouped_median.loc['female', 2, 'Miss']['Age']
    if sex == 'female' and pclass == 2 and title == 'Misses':
        return grouped_median.loc['female', 2, 'Misses']['Age']
    # Pclass of 3
    if sex == 'female' and pclass == 3 and title == 'Miss':
        return grouped_median.loc['female', 3, 'Miss']['Age']
    if sex == 'female' and pclass == 3 and title == 'Misses':
        return grouped_median.loc['female', 3, 'Misses']['Age']

    ### Males
    # Pclass of 1
    if sex == 'male' and pclass == 1 and title == 'Master':
        return grouped_median.loc['male', 1, 'Master']['Age']
    if sex == 'male' and pclass == 1 and title == 'Mister':
        return grouped_median.loc['male', 1, 'Mister']['Age']
    if sex == 'male' and pclass == 1 and title == 'Official':
        return grouped_median.loc['male', 1, 'Official']['Age']
    if sex == 'male' and pclass == 1 and title == 'Royalty':
        return grouped_median.loc['male', 1, 'Royalty']['Age']
    if sex == 'male' and pclass == 1 and title == 'Serveant':
        return grouped_median.loc['male', 1, 'Serveant']['Age']
    # Pclass of 2
    if sex == 'male' and pclass == 2 and title == 'Master':
        return grouped_median.loc['male', 2, 'Master']['Age']
    if sex == 'male' and pclass == 2 and title == 'Mister':
        return grouped_median.loc['male', 2, 'Mister']['Age']
    if sex == 'male' and pclass == 2 and title == 'Official':
        return grouped_median.loc['male', 2, 'Official']['Age']
    # Pclass of 3
    if sex == 'male' and pclass == 3 and title == 'Master':
        return grouped_median.loc['male', 3, 'Master']['Age']
    if sex == 'male' and pclass == 3 and title == 'Mister':
        return grouped_median.loc['male', 3, 'Mister']['Age']
    # Raise an error if we run into a scenario that's not listed above
    raise NotImplemented

def filter_age(train_length, data):
    group = ['Sex', 'Pclass', 'Title']

    grouped_train = data.head(train_length).groupby(group)
    grouped_median_train = grouped_train.median()
    grouped_test = data.iloc[train_length:].groupby(group)
    grouped_median_test = grouped_test.median()

    data.head(train_length).Age = data.head(train_length).apply(
        lambda r : replace_age(r, grouped_median_train) \
            if np.isnan(r['Age'])
            else r['Age'],
        axis=1
    )
    data.iloc[train_length:].Age = data.iloc[train_length:].apply(
        lambda r : replace_age(r, grouped_median_test) \
            if np.isnan(r['Age'])
            else r['Age'],
        axis=1
    )
    return data

def drop_name(data):
    return data.drop('Name', axis=1)

def encode_titles(data):
    dummy_titles = pd.get_dummies(data['Title'], prefix='Title')
    data = pd.concat([data, dummy_titles], axis=1)
    return data.drop('Title', axis=1)

def replace_missing_fare(train_length, data):
    data.head(train_length).Fare.fillna(
        data.head(train_length).Fare.mean(),
        inplace=True
    )
    data.iloc[train_length:].Fare.fillna(
        data.iloc[train_length:].Fare.mean(),
        inplace=True
    )
    return data

def replace_missing_embarked(train_length, data):
    # Default replace of 'S' which is the most common port
    data.head(train_length).Embarked.fillna('S', inplace=True)
    data.iloc[train_length:].Embarked.fillna('S', inplace=True)
    return data

def encode_embarked(data):
    dummy_embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    data = pd.concat([data, dummy_embarked], axis=1)
    return data.drop('Embarked', axis=1)

def process_cabin(data):
    # Fill the missing values with U for Unknown
    data.Cabin.fillna('U', inplace=True)
    data['Cabin'] = data['Cabin'].map(lambda c : c[0])
    dummy_cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
    data = pd.concat([data, dummy_cabin], axis=1)
    return data.drop('Cabin', axis=1)

def process_sex(data):
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
    return data

def process_pclass(data):
    dummy_pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')
    data = pd.concat([data, dummy_pclass], axis=1)
    return data.drop('Pclass', axis=1)

def filter_ticket(ticket):
    ticket = ticket.replace('.', '').replace('/', '').split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = filter(lambda t : not t.isdigit(), ticket)
    if len(ticket) > 0:
        return ticket[0]
    else:
        # If the ticket is numerical-only, return a default string
        return 'XXX'

def process_ticket(data):
    data['Ticket'] = data['Ticket'].map(filter_ticket)
    dummy_tickets = pd.get_dummies(data['Ticket'], prefix='Ticket')
    data = pd.concat([data, dummy_tickets], axis=1)
    return data.drop('Ticket', axis=1)

def process_family(data):
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
    data['Singleton'] = data['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    data['SmallFamily'] = data['FamilySize'].map(
        lambda s : 1 if 2 <= s <= 4 else 0
    )
    data['LargeFamily'] = data['FamilySize'].map(lambda s : 1 if s > 4 else 0)
    return data

def remove_passenger_id(data):
    return data.drop('PassengerId', axis=1)

def compute_score(clf, x, y, scoring='accuracy'):
    xval = cross_val_score(clf, x, y, cv=5, scoring=scoring)
    return np.mean(xval)

def save_prediction(data):
    data['PassengerId'] = data['PassengerId']

def main():
    train_data = read_csv_to_pandas_dataframe(TRAIN_FILE)
    test_data = read_csv_to_pandas_dataframe(TEST_FILE)

    targets = get_targets(train_data)

    combined = combine_dataframes(train_data, test_data)
    combined = extract_titles(combined)
    combined = filter_age(len(train_data), combined)
    combined = drop_name(combined)
    combined = encode_titles(combined)
    combined = replace_missing_fare(len(train_data), combined)
    combined = replace_missing_embarked(len(train_data), combined)
    combined = encode_embarked(combined)
    combined = process_cabin(combined)
    combined = process_sex(combined)
    combined = process_pclass(combined)
    combined = process_ticket(combined)
    combined = process_family(combined)
    combined = remove_passenger_id(combined)

    edited_train = combined.head(len(train_data))
    edited_test = combined.iloc[(len(train_data)):]

    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(edited_train, targets)

    features = pd.DataFrame()
    features['feature'] = edited_train.columns
    features['importance'] = clf.feature_importances_

    model = SelectFromModel(clf, prefit=True)
    train_reduced = model.transform(edited_train)
    test_reduced = model.transform(edited_test)

    parameters = {
        'bootstrap': False,
        'min_samples_leaf': 3,
        'n_estimators': 50,
        'min_samples_split': 10,
        'max_features': 'sqrt',
        'max_depth': 6
    }
    model = RandomForestClassifier(**parameters)
    model.fit(edited_train, targets)

    print 'Accuracy: %s' % \
        compute_score(model, edited_train, targets, scoring='accuracy')

    output = model.predict(edited_test).astype(int)
    dataframe = pd.DataFrame()
    dataframe['PassengerId'] = test_data['PassengerId']
    dataframe['Survived'] = output
    dataframe[['PassengerId', 'Survived']].to_csv('results.csv', index=False)


if __name__ == '__main__':
    main()
