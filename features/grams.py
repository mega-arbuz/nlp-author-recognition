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

import collections

from features.feature import Feature


class Unigram(Feature):

    def __init__(self, messages):
        self.unigrams = set()
        for message in messages:
            self.unigrams.update(list(message))

        self.unigrams = sorted(self.unigrams)

    def get_features(self, message):
        histogram = collections.Counter(message)
        return [histogram[ch] if ch in histogram else 0 for ch in self.unigrams]


class Ngram(Feature):

    def __init__(self, n, messages):
        self.ngrams = set()
        for message in messages:
            self.ngrams.update([message[i:i + n] for i in range(len(message) - n + 1)])

        self.ngrams = sorted(self.ngrams)
        self.n = n

    def get_features(self, message):
        histogram = collections.Counter([message[i:i + self.n] for i in range(len(message) - self.n + 1)])
        return [histogram[ngram] if ngram in histogram else 0 for ngram in self.ngrams]
