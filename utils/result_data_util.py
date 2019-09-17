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

from collections import defaultdict


def merge_result(res_dict_list):
    merged_dict = defaultdict(list)

    for users, res_dict in res_dict_list:
        for k, v in res_dict.items():
            merged_dict[k].append((users, v))

    for k, v in merged_dict.items():
        v.sort(key=lambda tup: tup[0])

    return merged_dict


def print_result(res_dict: dict):
    key_list = list(res_dict.keys())

    # Print title
    print('==============================')
    print('CSV Data:')
    print('==============================')
    print(', '.join(['Users'] + key_list))

    # Add first row with first values from tuples (num of users)
    rows = [[str(res[0]) for res in res_dict[key_list[0]]]]

    # Add all other rows with results
    for key in key_list:
        rows.append([str(res[1]) for res in res_dict[key]])

    # Rotate and print
    for row in zip(*rows):
        print(', '.join(row))
