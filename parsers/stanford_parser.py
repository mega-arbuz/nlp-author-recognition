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

import json
import os
import zipfile

import wget
from stanfordcorenlp import StanfordCoreNLP

from parsers.nlp_parser import NlpParser

CACHE_DIR = 'cache'


class CacheDict:
    filename: str
    cache: dict

    def __init__(self, filename):
        self.filename = filename
        try:
            with open(self.filename) as cache_file:
                self.cache = json.load(cache_file)
        except Exception:
            self.cache = dict()

    def close(self):
        with open(self.filename, 'w') as cache_file:
            json.dump(self.cache, cache_file)

    def __getitem__(self, key):
        return self.cache.get(key)

    def __setitem__(self, key, value):
        self.cache[key] = value


class StanfordParser(NlpParser):
    stanford_parser: StanfordCoreNLP = None
    dependency_cache: CacheDict = None
    constituency_cache: CacheDict = None

    def __init__(self, data_set_name, auto_start=True, is_cached=False):
        self._download_stanford_tools()
        self.is_cached = is_cached
        self.auto_start = auto_start
        if self.auto_start:
            self.stanford_parser = StanfordCoreNLP(r'en/stanford-corenlp-full-2018-10-05', memory='8g', timeout=30000)
        else:
            self.stanford_parser = StanfordCoreNLP(r'http://localhost:9001/', port=9001)

        # Cache
        if self.is_cached:
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR)
            self.dependency_cache = CacheDict(os.path.join(CACHE_DIR, f'{data_set_name}_dependency_cache.json'))
            self.constituency_cache = CacheDict(os.path.join(CACHE_DIR, f'{data_set_name}_constituency_cache.json'))

    def close(self):
        if self.auto_start:
            self.stanford_parser.close()

        if self.is_cached:
            self.dependency_cache.close()
            self.constituency_cache.close()

    def pos_tag(self, sentence):
        return self.stanford_parser.pos_tag(sentence)

    def parse(self, sentence):
        return self._execute_cached(self.constituency_cache,
                                    self.is_cached,
                                    self.stanford_parser.parse,
                                    sentence)

    def dependency_parse(self, sentence):
        return self._execute_cached(self.dependency_cache,
                                    self.is_cached,
                                    self.stanford_parser.dependency_parse,
                                    sentence)

    @staticmethod
    def _execute_cached(cache, is_cached, method, sentence):
        if is_cached and cache[sentence]:
            return cache[sentence]
        else:
            tree = method(sentence)
            if is_cached:
                cache[sentence] = tree
            return tree

    @staticmethod
    def _download_stanford_tools():
        stanford_core_nlp = 'stanford-corenlp-full-2018-10-05'
        file_name = stanford_core_nlp + '.zip'
        en_dir = 'en'
        zip_file = os.path.join(en_dir, file_name)
        stanford_dir = os.path.join(en_dir, stanford_core_nlp)

        if not os.path.isdir(en_dir):
            os.mkdir(en_dir)

        if not os.path.isdir(stanford_dir):
            if not os.path.isfile(zip_file):
                print('Beginning file download with wget module')
                url = 'http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip'
                wget.download(url, zip_file, bar=wget.bar_thermometer)
                print('download ended')
            with zipfile.ZipFile('en/stanford-corenlp-full-2018-10-05.zip', 'r') as zip_ref:
                zip_ref.extractall(en_dir)
