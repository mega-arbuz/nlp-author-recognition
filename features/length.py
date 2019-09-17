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

import operator
import statistics
from functools import reduce

import nltk

from features.feature import Feature


class MessageLength(Feature):

    def get_features(self, message):
        return [len(message)]


class SentenceLength(Feature):

    def get_features(self, message):
        sentences = reduce(operator.concat, map(nltk.sent_tokenize, message.splitlines()))
        sentences_length = list(map(len, sentences))

        return self._statistic_features(sentences_length)

    @staticmethod
    def _statistic_features(values):
        max_value = max(values)
        mean = statistics.mean(values)
        variance = statistics.variance(values) if len(values) > 1 else 0
        harmonic = statistics.harmonic_mean(values)
        median = statistics.median(values)
        median_high = statistics.median_high(values)
        return [max_value, mean, variance, harmonic, median, median_high]
