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

import argparse
import os
import traceback

from classifiers.classifier import Classifier
from classifiers.logistic_regression import LogisticRegressionClassifier
from features.constituency import Constituency
from features.dependency import Dependency
from features.grams import Unigram, Ngram
from features.length import SentenceLength, MessageLength
from features.pos import PartOfSpeechTags
from parsers.nlp_parser import NlpParser
from utils import csv_data_util, result_data_util
from utils.csv_data_util import ClassifierData
from features.features_vector import FeatureVector
from parsers.stanford_parser import StanfordParser
from utils.time_utils import Timer

# os.environ['PATH'] += ':/usr/lib/jvm/jdk1.8.0_121/bin'

DATA_MOVIES_120 = 'movies_120'
DATA_LEARN_PYTHON_500 = 'learn_python_500'
DATA_DND_500 = 'dnd_500'

VERBOSE = True

FEATURES_ALL = 'all'
FEATURES_COMBINED = 'combined'
FEATURES_SINGLES = 'singles'
FEATURES_GROUPS = 'groups'
FEATURES_LEXICAL = 'lexical'
FEATURES_SYNTACTIC = 'syntactic'
FEATURES_CONSTITUENCY = 'constituency'
FEATURES_POS_TAG = 'pos_tags'
FEATURES_DEPENDENCY = 'dependency'
FEATURES_SENTENCE_LENGTH = 'sentence_length'
FEATURES_MESSAGE_LENGTH = 'message_length'
FEATURES_UNIGRAM = 'unigram'
FEATURES_TRIGRAM = 'trigram'

FEATURES_CHOICES = [FEATURES_ALL, FEATURES_COMBINED, FEATURES_SINGLES, FEATURES_GROUPS,
                    FEATURES_LEXICAL, FEATURES_SYNTACTIC, FEATURES_CONSTITUENCY, FEATURES_POS_TAG,
                    FEATURES_DEPENDENCY, FEATURES_SENTENCE_LENGTH, FEATURES_MESSAGE_LENGTH,
                    FEATURES_UNIGRAM, FEATURES_TRIGRAM]
DATA_CHOICES = [DATA_MOVIES_120, DATA_LEARN_PYTHON_500, DATA_DND_500]


def main(no_auto_start: bool, not_cached: bool, data_set_name: str, features_type: str, users_min: int, users_max: int, num_iterations: int):
    auto_start = not no_auto_start
    cached = not not_cached
    # Initialize Stanford NLP
    nlp_parser = StanfordParser(data_set_name=data_set_name, auto_start=auto_start, is_cached=cached)

    # Run style recognition
    print(f'data: {data_set_name}')
    try:
        dict_list = list()
        for user_num in range(users_min, users_max + 1):
            for _ in range(num_iterations):
                # Load data
                data = csv_data_util.load_classifier_data(data_set_name=data_set_name, users_num=user_num, test_ratio=0.3)
                selected_features = get_features(nlp_parser, data, features_type)
                res_dict = analyze(LogisticRegressionClassifier(), data, selected_features)
                dict_list.append((user_num, res_dict))

        result = result_data_util.merge_result(dict_list)
        result_data_util.print_result(result)

    except Exception:
        print(traceback.format_exc())

    # Release parser
    nlp_parser.close()


def analyze(classifier: Classifier, data: ClassifierData, features):
    result_dict = dict()

    # Parse and recognize style
    for feature in features:
        res = train_test(feature, classifier, data=data)
        result_dict[feature.name] = res
        print(f'feature group: {feature.name}, f1-score: {res}')

    return result_dict


def train_test(features_vector: FeatureVector, classifier: Classifier, data: ClassifierData):
    # Build train features vector
    with Timer('building train features', VERBOSE):
        x_train_features = features_vector.convert_to_features(data.x_train, VERBOSE)

    # Train
    with Timer('training', VERBOSE):
        classifier.train(x_train_features, data.y_train)

    # Build test features vector
    with Timer('building test features', VERBOSE):
        x_test_features = features_vector.convert_to_features(data.x_test, VERBOSE)

    # Test
    if VERBOSE:
        print(features_vector.name)
        print(classifier.report(x_test_features, data.y_test))

    return classifier.f1_micro(x_test_features, data.y_test)


def get_features(nlp_parser: NlpParser, data: ClassifierData, features: str):
    # Initialize features
    if features == FEATURES_ALL:
        return [
            FeatureVector('All',
                          Constituency(nlp_parser),
                          PartOfSpeechTags(nlp_parser),
                          Dependency(nlp_parser),
                          SentenceLength(),
                          MessageLength(),
                          Unigram(data.x_train),
                          Ngram(3, data.x_train))
        ]

    if features == FEATURES_COMBINED:
        return [
            FeatureVector('Combined', Dependency(nlp_parser), Constituency(nlp_parser),
                          PartOfSpeechTags(nlp_parser), Unigram(data.x_train), Ngram(3, data.x_train))
        ]

    if features == FEATURES_LEXICAL:
        return [
            FeatureVector('Lexical', Unigram(data.x_train), Ngram(3, data.x_train), SentenceLength())
        ]

    if features == FEATURES_SYNTACTIC:
        return [
            FeatureVector('Syntactic', Constituency(nlp_parser), PartOfSpeechTags(nlp_parser), Dependency(nlp_parser))
        ]

    if features == FEATURES_CONSTITUENCY:
        return [FeatureVector('Constituency', Constituency(nlp_parser))]

    if features == FEATURES_POS_TAG:
        return [FeatureVector('POS Tags', PartOfSpeechTags(nlp_parser))]

    if features == FEATURES_DEPENDENCY:
        return [FeatureVector('Dependency', Dependency(nlp_parser))]

    if features == FEATURES_SENTENCE_LENGTH:
        return [FeatureVector('Sentence Length', SentenceLength())]

    if features == FEATURES_MESSAGE_LENGTH:
        return [FeatureVector('Message Length', MessageLength())]

    if features == FEATURES_UNIGRAM:
        return [FeatureVector('Unigram', Unigram(data.x_train))]

    if features == FEATURES_TRIGRAM:
        return [FeatureVector('Trigram', Ngram(3, data.x_train))]

    if features == FEATURES_SINGLES:
        return get_features(nlp_parser, data, FEATURES_CONSTITUENCY) + \
               get_features(nlp_parser, data, FEATURES_POS_TAG) + \
               get_features(nlp_parser, data, FEATURES_DEPENDENCY) + \
               get_features(nlp_parser, data, FEATURES_SENTENCE_LENGTH) + \
               get_features(nlp_parser, data, FEATURES_MESSAGE_LENGTH) + \
               get_features(nlp_parser, data, FEATURES_UNIGRAM) + \
               get_features(nlp_parser, data, FEATURES_TRIGRAM)

    if features == FEATURES_GROUPS:
        return get_features(nlp_parser, data, FEATURES_ALL) + \
               get_features(nlp_parser, data, FEATURES_COMBINED) + \
               get_features(nlp_parser, data, FEATURES_LEXICAL) + \
               get_features(nlp_parser, data, FEATURES_SYNTACTIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-na', dest='no_auto_start', help='connect to a running nlp engine instead of starting a new server',
                        required=False, action='store_true', default=False)
    parser.add_argument('-nc', dest='no_cache', help='don\'t cache results from nlp engine', required=False, action='store_true', default=False)
    parser.add_argument('-s', dest='silent', help='non verbose print (silent)', required=False, action='store_true', default=False)
    parser.add_argument('-d', dest='data_name', help='dataset name', required=False, default=DATA_MOVIES_120, choices=DATA_CHOICES)
    parser.add_argument('-i', dest='iterations', help='number of iterations (per feature)', type=int, required=False, default=1)
    parser.add_argument('-umin', dest='users_min', help='minimum number of users', type=int, required=False, default=10)
    parser.add_argument('-umax', dest='users_max', help='maximum number of users', type=int, required=False, default=10)
    parser.add_argument('-f', dest='features', help='feature set', required=False, default=FEATURES_COMBINED, choices=FEATURES_CHOICES)

    options = parser.parse_args()

    VERBOSE = not options.silent

    main(options.no_auto_start, options.no_cache, options.data_name, options.features, options.users_min, options.users_max, options.iterations)
