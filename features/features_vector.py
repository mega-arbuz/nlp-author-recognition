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
from functools import reduce
from multiprocessing.pool import ThreadPool

from tqdm import tqdm


class FeatureVector:

    def __init__(self, name, *features):
        self.name = name
        self.features = list(features)

    def convert_to_features(self, data: list, verbose):
        pool = ThreadPool(8)
        if verbose:
            features = list(tqdm(pool.imap(self._build_vector, data), total=len(data)))
        else:
            features = list(pool.map(self._build_vector, data))
        pool.close()

        return features

    def _build_vector(self, data_element):
        return reduce(operator.concat, map(lambda feature: feature.get_features(data_element), self.features))
