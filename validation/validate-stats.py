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
import logging

import pandas as pd
import tensorflow_data_validation as tfdv
from google.protobuf.json_format import MessageToDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-t',
                        '--stats_file_1',
                        type=str,
                        required=False,
                        default='./train_stats/stats.txt',
                        help='Path to the training/reference stats .txt file ')

    parser.add_argument('-v',
                        '--stats_file_2',
                        type=str,
                        required=False,
                        default='./val_stats/stats.txt',
                        help='Path to the validation stats .txt file ')


    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info(f"Args: {args}")


    stats1 = tfdv.load_stats_text(input_path=args.stats_file_1)
    stats2 = tfdv.load_stats_text(input_path=args.stats_file_2)

    schema1 = tfdv.infer_schema(statistics=stats1)

    # Custom rules, tweak this as required. This is just an example
    tfdv.get_feature(schema1, 'I1').drift_comparator.jensen_shannon_divergence.threshold = 0.06

    # Calculate drift between the reference stats stats1, and the statistics from new data in stats2
    drift_anomalies = tfdv.validate_statistics(statistics=stats2,
                                                schema=schema1,
                                                previous_statistics=stats1)

    # Convert the .pb2 to dict
    drift = MessageToDict(drift_anomalies)

    value = drift['driftSkewInfo'][0]['driftMeasurements'][0]['value']
    threshold = drift['driftSkewInfo'][0]['driftMeasurements'][0]['threshold']
    logging.info(f"JS divergence value: {value}, and JS divergence threshold: {threshold}")
    drift_detected = True
    if value < threshold:
        drift_detected = False
    logging.info(f"Drift detected: {drift_detected}")
