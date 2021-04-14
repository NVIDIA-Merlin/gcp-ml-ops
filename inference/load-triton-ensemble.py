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



import sys
import argparse
import logging

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


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

    parser.add_argument('-v',
                        '--verbose',
                        type=bool,
                        required=False,
                        default=True,
                        help='Verbosity, True or False')


    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info(f"Args: {args}")
    
    try:
        triton_client = grpcclient.InferenceServerClient(url=args.triton_grpc_url, verbose=args.verbose)
        logging.info("Triton client created.")
    except Exception as e:
        logging.error(f"channel creation failed: {str(e)}")
        sys.exit()


    # Health
    if not triton_client.is_server_live(headers={'test': '1', 'dummy': '2'}):
        logging.error("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        logging.error("FAILED : is_server_ready")
        sys.exit(1)

    logging.info(f"Models available: {triton_client.get_model_repository_index()}")

    # Load the ensemble model
    # TODO: Increase the timeout. Sometimes this times out with 8xGPUs because loading
    # the model takes longer.
    try:
        triton_client.load_model(model_name=args.model_name)
    except InferenceServerException as e:
        if "failed to load" in e.message():
            logging.error(f"Model {args.model_name} failed to load!")

    if not triton_client.is_model_ready(args.model_name):
        logging.error(f"Model {args.model_name} is not ready!")
        sys.exit(1)
    else:
        logging.info(f"Model {args.model_name} is ready!")
