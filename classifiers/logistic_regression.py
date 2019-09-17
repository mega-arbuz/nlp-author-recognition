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

from classifiers.classifier import Classifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics


class LogisticRegressionClassifier(Classifier):

    model = None

    def __init__(self):
        self.model = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=1000, n_jobs=1)

    def train(self, x_data, y_data):
        self.model.fit(x_data, y_data)

    def report(self, x_data, y_data):
        y_prediction = self.model.predict(x_data)
        return sklearn.metrics.classification_report(y_data, y_prediction)

    def f1_micro(self, x_data, y_data):
        y_prediction = self.model.predict(x_data)
        return sklearn.metrics.f1_score(y_data, y_prediction, average='micro')
