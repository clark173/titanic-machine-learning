import constants as c
import csv
import pandas as pd
import tensorflow as tf
from passengers import Passengers


class Features:
    def __init__(self):
        self._saved_passenger_id_list = []

    def analyze(self):
        train_passengers = self._get_passengers(c.TRAIN_FILE, predict=False)
        test_passengers = self._get_passengers(c.TEST_FILE, predict=True)
        train_no_age_passengers = self._get_passengers(c.TRAIN_FILE,
                                                       predict=False,
                                                       omit_age=True)
        test_no_age_passengers = self._get_passengers(c.TEST_FILE,
                                                      predict=True,
                                                      omit_age=True)
        feature_columns = self._build_feature_columns(c.COLUMN_X_DATA_SET)
        features_no_age = self._build_feature_columns(c.COLUMN_X_DATA_SET_NO_AGE)
        classifier = self._create_classifier(feature_columns,
                                             train_passengers,
                                             test_passengers,
                                             c.COLUMN_X_DATA_SET,
                                             'model_data')
        classifier_no_age = self._create_classifier(features_no_age,
                                                    train_no_age_passengers,
                                                    test_no_age_passengers,
                                                    c.COLUMN_X_DATA_SET_NO_AGE,
                                                    'model_data_no_age')

    def _get_passengers(self, file_to_read, predict=False, omit_age=False):
        return Passengers(file_to_read, predict, omit_age)

    def _build_feature_columns(self, data_set):
        shape = len(data_set)
        return [tf.feature_column.numeric_column('x', shape=[shape])]

    def _input_fn(self, passengers, data_set, predict=False):
        if predict:
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': passengers.data_set[data_set].as_matrix()},
                num_epochs=1,
                shuffle=False
            )
        else:
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': passengers.data_set[data_set].as_matrix()},
                y=passengers.data_set[c.COLUMN_Y_DATA_SET].as_matrix(),
                num_epochs=1,
                shuffle=True
            )
        return input_fn

    def _prediction(self, prediction_data):
        died, survived = prediction_data['probabilities']
        if survived > died:
            return c.SURVIVED
        return c.DIED

    def _create_csv(self, results_list):
        with open('results.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['PassengerId', 'Survived'])
            for row in results_list:
                writer.writerow(row)

    def _build_predictions_file(self, predictions, test_passengers):
        results_list = []
        passengerid = 0
        for prediction in predictions:
            if test_passengers.passenger_id_list[passengerid] in self._passenger_id_list:
                continue
            result = self._prediction(prediction)
            results_list.append([test_passengers.passenger_id_list[passengerid],
                                 result])
            self._passenger_id_list.append(test_passengers.passenger_id_list[passengerid])
            passengerid += 1
        print results_list

    def _create_classifier(self, feature_columns, train_passengers, test_passengers, data_set, model_dir):
        classifier = tf.estimator.DNNClassifier(
                                                feature_columns=feature_columns,
                                                hidden_units=[100, 200, 100],
                                                n_classes=2,
                                                model_dir=model_dir
                                               )
        train_input_fn = self._input_fn(train_passengers, data_set, predict=False)
        predict_input_fn = self._input_fn(test_passengers, data_set, predict=True)
        classifier.train(input_fn=train_input_fn, steps=700)
        print classifier.evaluate(input_fn=train_input_fn)['accuracy'] * 100.0
        predictions = list(classifier.predict(input_fn=predict_input_fn))
        self._build_predictions_file(predictions, test_passengers)
        self._create_csv(results_list)
