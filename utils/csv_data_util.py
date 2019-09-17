#  Copyright (C) 2019 Oleg Shnaydman, Victoria Smolensky
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import csv
import random


class ClassifierData:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


def load_classifier_data(data_set_name, users_num, test_ratio, posts_num=-1) -> ClassifierData:
    with open('data/{}.csv'.format(data_set_name)) as csv_file:

        file_dict = dict()

        x_data_train = list()
        y_data_train = list()
        x_data_test = list()
        y_data_test = list()

        csv_reader = csv.reader(csv_file, delimiter=',')

        # Load all data into dictionary (user_id, message)
        for row in csv_reader:
            current_id = row[0]
            if file_dict.get(current_id):
                file_dict.get(current_id).append(row[1])
            else:
                file_dict[current_id] = [row[1]]

        # Randomize messages
        for user_id, posts_list in file_dict.items():
            if posts_num < 0:
                posts_num = len(posts_list)
            if len(posts_list) < posts_num:
                raise IndexError('data is smaller than requested length')
            random.shuffle(posts_list)

        test_size = round(posts_num * test_ratio)
        train_size = posts_num - test_size

        # Randomize users
        user_ids = list(file_dict.items())
        random.shuffle(user_ids)

        # Select train and test data
        for user_id, posts_list in user_ids[:users_num]:
            x_data_train.extend(posts_list[:train_size])
            y_data_train.extend(train_size * [user_id])
            x_data_test.extend(posts_list[-test_size:])
            y_data_test.extend(test_size * [user_id])

        return ClassifierData(x_data_train, y_data_train, x_data_test, y_data_test)



