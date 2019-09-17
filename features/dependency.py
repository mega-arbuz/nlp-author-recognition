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
import operator
import statistics
from functools import reduce

import nltk

from features.feature import Feature
from parsers.nlp_parser import NlpParser


class Dependency(Feature):
    MODS = ['ROOT', 'acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'auxpass',
            'case', 'cc', 'cc:preconj', 'ccomp', 'compound', 'compound:prt', 'conj', 'cop',
            'csubj', 'csubjpass', 'dep', 'det', 'det:predet', 'discourse', 'dobj', 'expl',
            'iobj', 'mark', 'mwe', 'neg', 'nmod', 'nmod:npmod', 'nmod:poss', 'nmod:tmod',
            'nsubj', 'nsubjpass', 'nummod', 'parataxis', 'punct', 'root', 'xcomp']

    def __init__(self, stanford_parser: NlpParser):
        self.nlp_parser = stanford_parser
        nltk.download('punkt')

    def get_features(self, message):
        try:
            # Convert message to a list of separated sentences -
            # it will be quicker to analyze separate sentences with nlp engine.
            sentences = reduce(operator.concat, map(nltk.sent_tokenize, message.splitlines()))

            try:
                dependency_tree = self.nlp_parser.dependency_parse(message)
            except Exception:
                dependency_tree = reduce(operator.concat, map(self.nlp_parser.dependency_parse, sentences))

            # Find all indices with ROOT element
            root_indices = [i for i, (mod, _, _) in enumerate(dependency_tree) if mod == 'ROOT'] + [len(dependency_tree)]
            # Split result to sentences with ROOT element at the beginning
            sentences = [dependency_tree[s:e] for s, e in zip(root_indices[:-1], root_indices[1:])]

            root_distance = list()
            root_percentage = list()
            sentence_max_distance = list()
            sentence_max_distance_percentage = list()
            root_children = list()
            root_children_percentage = list()
            histogram_mods = collections.Counter()
            for sentence in sentences:
                # Root distance
                mod, parent, root_index = sentence[0]
                distance = abs(parent - root_index)
                root_distance.append(distance)
                root_percentage.append(int(100 * (distance / len(sentence))))

                # Root children
                root_children_count = sum([1 for _, parent, _ in sentence if parent == root_index])
                root_children.append(root_children_count)
                root_children_percentage.append(int(100 * (root_children_count / len(sentence))))

                # Sentence
                max_distance = max([abs(parent - index) for _, parent, index in sentence])
                sentence_max_distance.append(max_distance)
                sentence_max_distance_percentage.append(int(100 * (max_distance / len(sentence))))

                # Count mods
                histogram_mods.update([mod for mod, _, _ in sentence])

            histogram_mods_sparse = [histogram_mods[mod] if mod in histogram_mods else 0 for mod in self.MODS]

            return histogram_mods_sparse + reduce(operator.concat,
                                                  map(self._statistic_features,
                                                      [root_distance, root_percentage,
                                                       sentence_max_distance, sentence_max_distance_percentage,
                                                       root_children, root_children_percentage]))
        except Exception:
            return (len(self.MODS) + len(self._statistic_features([1, 2, 3])) * 6) * [0]

    @staticmethod
    def _statistic_features(values):
        max_value = max(values)
        mean = statistics.mean(values)
        variance = statistics.variance(values) if len(values) > 1 else 0
        harmonic = statistics.harmonic_mean(values)
        median_high = statistics.median_high(values)
        return [max_value, mean, variance, harmonic, median_high]
