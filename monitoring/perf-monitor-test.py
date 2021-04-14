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

import numpy as np
import os
import logging
import argparse
import sys
import warnings
import sys
import time
import json

import cudf
from sklearn import metrics
import pandas as pd

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

from google.cloud import pubsub_v1
from google.protobuf.json_format import MessageToJson
from google.pubsub_v1.types import Encoding



def publish_batch(project_id, topic_id, current_batch, pred_label):
    # Initialize a Publisher client.
    client = pubsub_v1.PublisherClient()
    topic_path = client.topic_path(project_id, topic_id)

    batch_size = len(pred_label)
    df = current_batch.to_pandas()

    for i in range(batch_size):
        row = df.iloc[i]

        frame = {
            "input0": row[CONTINUOUS_COLUMNS].values.tolist(),
            "input1": row[CATEGORICAL_COLUMNS].values.tolist(),
            "trueval": row['label'],
            "predval": response.as_numpy("OUTPUT0")[i].astype('float64')
        }

        payload = json.dumps(frame).encode('utf-8')

        # When you publish a message, the client returns a future.
        api_future = client.publish(topic_path, data=''.encode(), payload=payload)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-u',
                        '--triton_grpc_url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='URL to Triton gRPC Endpoint')

    parser.add_argument('-m',
                        '--model_name',
                        type=str,
                        required=False,
                        default='dcn_ens',
                        help='Name of the model ensemble to load')

    parser.add_argument('-d',
                        '--test_data',
                        type=str,
                        required=False,
                        default='/crit_int_pq/day_23.parquet',
                        help='Path to a test .parquet file. Default')

    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=False,
                        default=64,
                        help='Batch size. Max is 64 at the moment, but this max size could be specified when create the model and the ensemble.')

    parser.add_argument('-n',
                        '--n_batches',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of batches of data to send')

    parser.add_argument('-v',
                        '--verbose',
                        type=bool,
                        required=False,
                        default=False,
                        help='Verbosity, True or False')

    parser.add_argument("--project_id",
                        type=str,
                        required=True,
                        default="dl-tme",
                        help="Google Cloud project ID")

    parser.add_argument("--topic_id",
                        type=str,
                        required=True,
                        default="pubsub",
                        help="Pub/Sub topic ID")


    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info(f"Args: {args}")


    # warnings can be disabled
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    try:
        triton_client = grpcclient.InferenceServerClient(url=args.triton_grpc_url, verbose=args.verbose)
        logging.info("Triton client created.")

        triton_client.is_model_ready(args.model_name)
        logging.info(f"Model {args.model_name} is ready!")
    except Exception as e:
        logging.error(f"Channel creation failed:  {str(e)}")
        sys.exit()

    # Load the dataset
    CATEGORICAL_COLUMNS =  ['C' + str(x) for x in range(1,27)]
    CONTINUOUS_COLUMNS = ['I' + str(x) for x in range(1,14)]
    LABEL_COLUMNS = ['label']
    col_names =  CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS
    col_dtypes = [np.int32]*26 + [np.int64]*13

    logging.info("Reading dataset..")
    all_batches = cudf.read_parquet(args.test_data, num_rows=args.batch_size*args.n_batches)

    results=[]

    with grpcclient.InferenceServerClient(url=args.triton_grpc_url) as client:
        for batch in range(args.n_batches):

            logging.info(f"Requesting inference for batch {batch}..")
            start_idx = batch*args.batch_size
            end_idx = (batch+1)*(args.batch_size)

            # Convert the batch to a triton inputs
            current_batch = all_batches[start_idx:end_idx]
            columns = [(col, current_batch[col]) for col in col_names]
            inputs = []

            for i, (name, col) in enumerate(columns):
                d = col.values_host.astype(col_dtypes[i])
                d = d.reshape(len(d), 1)
                inputs.append(grpcclient.InferInput(name, d.shape, np_to_triton_dtype(col_dtypes[i])))
                inputs[i].set_data_from_numpy(d)

            outputs = []
            outputs.append(grpcclient.InferRequestedOutput("OUTPUT0"))

            response = client.infer(args.model_name, inputs, request_id=str(1), outputs=outputs)

            results.extend(response.as_numpy("OUTPUT0"))

            publish_batch(args.project_id, args.topic_id,
                        current_batch,
                        response.as_numpy("OUTPUT0"))

    logging.info(f"ROC AUC Score: {metrics.roc_auc_score(all_batches[LABEL_COLUMNS].values.tolist(), results)}")