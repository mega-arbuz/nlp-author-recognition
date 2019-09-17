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
from parsers.nlp_parser import NlpParser


class PartOfSpeechTags(Feature):

    POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
                'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',
                'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

    def __init__(self, stanford_parser: NlpParser):
        self.nlp_parser = stanford_parser

    def get_features(self, message):
        tags = [y for x, y in self.nlp_parser.pos_tag(message)]
        histogram = collections.Counter(tags)
        return [histogram[tag] if tag in histogram else 0 for tag in self.POS_TAGS]
