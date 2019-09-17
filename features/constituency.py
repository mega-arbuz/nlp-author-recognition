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
import collections
import statistics
from functools import reduce

import nltk
from nltk import Tree

from features.feature import Feature
from parsers.nlp_parser import NlpParser


class Constituency(Feature):
    TAGS = ['ADJP', '-ADV', 'ADVP', '-BNF', 'CC', 'CD', '-CLF', '-CLR', 'CONJP', '-DIR', 'DT', '-DTV', 'EX',
            '-EXT', 'FRAG', 'FW', '-HLN', 'IN', 'INTJ', 'JJ', 'JJR', 'JJS', '-LGS', '-LOC', 'LS', 'LST', 'MD',
            '-MNR', 'NAC', 'NN', 'NNS', 'NNP', 'NNPS', '-NOM', 'NP', 'NX', 'PDT', 'POS', 'PP', '-PRD', 'PRN',
            'PRP', '-PRP', 'PRP$', 'PRT', '-PUT', 'QP', 'RB', 'RBR', 'RBS', 'RP', 'RRC', 'S', 'SBAR', 'SBARQ',
            '-SBJ', 'SINV', 'SQ', 'SYM', '-TMP', 'TO', '-TPC', '-TTL', 'UCP', 'UH', 'VB', 'VBD', 'VBG', 'VBN',
            'VBP', 'VBZ', '-VOC', 'VP', 'WDT', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'WP', 'WP$', 'WRB', 'X']

    def __init__(self, stanford_parser: NlpParser):
        self.nlp_parser = stanford_parser
        nltk.download('punkt')

    def get_features(self, message):
        # noinspection PyProtectedMember
        try:
            # Convert message to a list of separated sentences -
            # it will be quicker to analyze separate sentences with nlp engine.
            sentences = reduce(operator.concat, map(nltk.sent_tokenize, message.splitlines()))
            # Filter out long sentences - nlp engine may give timeout exception for long input.
            sentences = list(filter(lambda sentence: len(nltk.word_tokenize(sentence)) < 70, sentences))
            # Convert all constituency string trees to nltk Tree.
            trees = list(map(self.nlp_parser.parse, sentences))

            # Create lists of sentences depth and width.
            depth_list = list(map(self._calc_depth, trees))
            width_list = list(map(self._calc_width, trees))

            depth_percentage = list(map(self._cal_depth_percentage, trees))
            width_percentage = list(map(self._calc_width_percentage, trees))

            # Count tags
            histogram_tags = collections.Counter()
            histogram_tags_width = collections.Counter()
            for tree in trees:
                histogram_tags.update(self._tags_count(tree))
                histogram_tags_width.update(self._tag_width_count(tree))

            histogram_tags_sparse = [histogram_tags[tag] for tag in self.TAGS]
            histogram_tags_sparse_width = [histogram_tags_width[tag] for tag in self.TAGS]

            return histogram_tags_sparse + histogram_tags_sparse_width + self._statistic_features(depth_list) + self._statistic_features(width_list) + \
                self._statistic_features(depth_percentage) + self._statistic_features(width_percentage)
        except Exception:
            # print(traceback.format_exc())
            return (len(self.TAGS) * 2 + len(self._statistic_features([1, 2, 3])) * 4) * [0]

    @staticmethod
    def _statistic_features(values):
        max_value = max(values)
        mean = statistics.mean(values)
        variance = statistics.variance(values) if len(values) > 1 else 0
        harmonic = statistics.harmonic_mean(values)
        median = statistics.median(values)
        median_high = statistics.median_high(values)
        return [max_value, mean, variance, harmonic, median, median_high]

    def _tag_width_count(self, tree_str):
        nltk_tree = Tree.fromstring(tree_str)
        node_queue = list()
        node_queue.append((0, nltk_tree))

        tags_histogram = dict.fromkeys(self.TAGS, 0)
        while node_queue:
            depth, node = node_queue.pop(0)
            for child in node:
                if isinstance(child, Tree):
                    if child.label() in tags_histogram:
                        tags_histogram[child.label()] += len(child)
                    node_queue.append((depth, child))
        return tags_histogram

    def _tags_count(self, tree_str):
        nltk_tree = Tree.fromstring(tree_str)
        node_queue = list()
        node_queue.append((0, nltk_tree))

        tags_histogram = dict.fromkeys(self.TAGS, 0)
        while node_queue:
            depth, node = node_queue.pop(0)
            for child in node:
                if isinstance(child, Tree):
                    if child.label() in tags_histogram:
                        tags_histogram[child.label()] = 1
                    node_queue.append((depth, child))

        return tags_histogram

    @staticmethod
    def _calc_depth(tree_str):
        nltk_tree = Tree.fromstring(tree_str)
        node_queue = list()
        node_queue.append((0, nltk_tree))

        tree_depth = 0
        while node_queue:
            depth, node = node_queue.pop(0)
            depth += 1
            if depth > tree_depth:
                tree_depth = depth
            for child in node:
                if isinstance(child, Tree):
                    node_queue.append((depth, child))

        return tree_depth

    @staticmethod
    def _calc_width(tree_str):
        nltk_tree = Tree.fromstring(tree_str)
        return len(nltk_tree[0])

    @staticmethod
    def _cal_depth_percentage(tree_str):
        return 100 * (Constituency._calc_depth(tree_str) / Constituency._sentence_length(tree_str))

    @staticmethod
    def _calc_width_percentage(tree_str):
        return 100 * (Constituency._calc_width(tree_str) / Constituency._sentence_length(tree_str))

    @staticmethod
    def _sentence_length(tree_str):
        return len(Tree.fromstring(tree_str).flatten())
