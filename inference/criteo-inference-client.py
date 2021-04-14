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
import argparse
import sys
import warnings
import sys

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import *
import cudf

from sklearn import metrics


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


    args = parser.parse_args()

    # warnings can be disabled
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    try:
        triton_client = grpcclient.InferenceServerClient(url=args.triton_grpc_url, verbose=args.verbose)
        print("Triton client created.")
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()


    if not triton_client.is_model_ready(args.model_name):
        print(f"Model {args.model_name} is not ready!")
        sys.exit(1)
    else:
        print(f"Model {args.model_name} is ready!")

    ### ....
   

    # Load the dataset
    CATEGORICAL_COLUMNS =  ['C' + str(x) for x in range(1,27)]
    CONTINUOUS_COLUMNS = ['I' + str(x) for x in range(1,14)]
    LABEL_COLUMNS = ['label']
    col_names =  CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS
    col_dtypes = [np.int32]*26 + [np.int64]*13
    


    print("Reading dataset..")
    batch_whole = cudf.read_parquet(args.test_data, num_rows=args.batch_size*args.n_batches)
    batch_features = batch_whole[col_names]
    batch_labels = batch_whole[LABEL_COLUMNS]

    

    results=[]


    with grpcclient.InferenceServerClient(url=args.triton_grpc_url) as client:
        for batch in range(args.n_batches):
            print(f"Requesting inference for batch {batch}..")
            start_idx=batch*args.batch_size
            end_idx=(batch+1)*(args.batch_size)
            # convert the batch to a triton inputs
            columns = [(col, batch_features[col][start_idx:end_idx]) for col in col_names]
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
    

    print(f"ROC AUC Score: {metrics.roc_auc_score(batch_labels[LABEL_COLUMNS].values.tolist(), results)}")

    

