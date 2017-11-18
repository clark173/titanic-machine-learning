import pandas as pd
import numpy as np


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

def main():
    pd.options.display.max_columns = 100
    pd.options.display.max_rows = 100

    train_data = read_csv_to_pandas_dataframe(TRAIN_FILE)
    test_data = read_csv_to_pandas_dataframe(TEST_FILE)

    targets = get_targets(train_data)

    combined = combine_dataframes(train_data, test_data)
    combined = extract_titles(combined)
    combined = filter_age(len(train_data), combined)

if __name__ == '__main__':
    main()
