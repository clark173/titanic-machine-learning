import constants as c
import csv
import pandas as pd
import tensorflow as tf
from passengers import Passengers


class Features:
    def analyze(self):
        train_passengers = self._get_passengers(c.TRAIN_FILE, predict=False)
        test_passengers = self._get_passengers(c.TEST_FILE, predict=True)
        feature_columns = self._build_feature_columns()
        self._create_classifier(feature_columns, train_passengers, test_passengers)

    def _get_passengers(self, file_to_read, predict=False):
        return Passengers(file_to_read, predict)

    def _build_feature_columns(self):
        shape = len(c.COLUMN_X_DATA_SET)
        return [tf.feature_column.numeric_column('x', shape=[shape])]

    def _input_fn(self, passengers, predict=False):
        if predict:
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': passengers.data_set[c.COLUMN_X_DATA_SET].as_matrix()},
                num_epochs=1,
                shuffle=False
            )
        else:
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': passengers.data_set[c.COLUMN_X_DATA_SET].as_matrix()},
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

    def _create_classifier(self, feature_columns, train_passengers, test_passengers):
        classifier = tf.estimator.DNNClassifier(
                                                feature_columns=feature_columns,
                                                hidden_units=[100, 200, 100],
                                                n_classes=2,
                                                model_dir='model_data'
                                               )
        train_input_fn = self._input_fn(train_passengers, predict=False)
        predict_input_fn = self._input_fn(test_passengers, predict=True)
        classifier.train(input_fn=train_input_fn, steps=700)
        print classifier.evaluate(input_fn=train_input_fn)
        predictions = list(classifier.predict(input_fn=predict_input_fn))
        results_list = []
        passengerid = 0
        for prediction in predictions:
            result = self._prediction(prediction)
            results_list.append([test_passengers.passenger_id_list[passengerid],
                                 result])
            passengerid += 1
        self._create_csv(results_list)
