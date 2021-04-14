# Copyright (c) 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



import os
import argparse

import pandas as pd
import tensorflow_data_validation as tfdv
from google.protobuf.json_format import MessageToDict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        '--data_dir',
                        type=str,
                        required=False,
                        default='/crit_int_pq/day_23.parquet',
                        help='Path to a data .parquet file. Default')

    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=False,
                        default='./output',
                        help='Path to a where stats must be saved')

    parser.add_argument('-f',
                        '--file_name',
                        type=str,
                        required=False,
                        default='stats.txt',
                        help='Name of the stats file')


    args = parser.parse_args()


    # tfdv doesnt support generating stats directly from parquet
    # so read through pandas parquet reader
    # Ideally, this should be be an accelerated parquet reader and stats
    # computation should happen via GPU
    df = pd.read_parquet(args.data_dir)

    stats = tfdv.generate_statistics_from_dataframe(df)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, args.file_name)

    tfdv.write_stats_text(stats, output_path=output_path)
