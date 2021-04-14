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
import json

import nvtabular as nvt
from nvtabular.inference.triton import export_hugectr_ensemble
from nvtabular.ops import get_embedding_sizes


def create_ensemble(workflow_path, hugectr_model_path, ensemble_output_path, ensemble_config_file):
    """
    Creates an ensemble of NVTabular and HugeCTR model.

    This enables preprocessing at the time of inference, allowing the
    user to send raw data directly to the inference server.
    """

    # Load the workflow object
    workflow = nvt.Workflow.load(workflow_path)

    # Verify that the workflow is loaded
    embeddings = get_embedding_sizes(workflow)
    logging.info(f"Embedding sizes for categorical features: {embeddings}")

    with open(ensemble_config_file, "r") as jsonfile:
        ensemble_config = json.load(jsonfile)

    hugectr_params = ensemble_config["hugectr_params"]

    # We override the config param to update the model version
    # Get the model version for updating the config accordingly
    model_version = hugectr_model_path.split('/')[-2]
    logging.info(f"Model version: {model_version}")
    model_json_path = hugectr_params["config"].split(os.sep) # "/model/models/dcn/1/dcn.json" -> ['', 'model', 'models', 'dcn', '1', 'dcn.json']
    model_json_path[-2] = model_version # ['', 'model', 'models', 'dcn', '1', 'dcn.json'] -> ['', 'model', 'models', 'dcn', '2', 'dcn.json']
    hugectr_params["config"] = os.sep + os.path.join(*model_json_path) # '/' + 'model/models/dcn/2/dcn.json'

    logging.info(f"HugeCTR configs: {hugectr_params}")

    categorical_cols =  ensemble_config["categorical_cols"]
    continuous_cols = ensemble_config["continuous_cols"]
    label_cols = ensemble_config["label_cols"]

    logging.info(f"Categorical Columns: {categorical_cols}")
    logging.info(f"Continuous Columns: {continuous_cols}")
    logging.info(f"Label Columns: {label_cols}")

    logging.info(f"Generating the ensemble at directory: {ensemble_output_path}")
    export_hugectr_ensemble(workflow=workflow,
                            hugectr_model_path=hugectr_model_path,
                            hugectr_params=hugectr_params,
                            name=ensemble_config["name"],
                            output_path=ensemble_output_path,
                            label_columns=label_cols,
                            cats=categorical_cols,
                            conts=continuous_cols,
                            max_batch_size=ensemble_config["max_batch_size"])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--nvt_workflow_path',
                        type=str,
                        required=False,
                        default='./test_dask/output/workflow',
                        help='Path to Workflow Dir. Default is ./test_dask/output/workflow')

    parser.add_argument('-m',
                        '--hugectr_model_path',
                        type=str,
                        required=False,
                        default='/model/criteo_hugectr/1/',
                        help='Path to where your .model files and inference .json is stored. Default is /model/criteo_hugectr/1/')

    parser.add_argument('-o',
                        '--ensemble_output_path',
                        type=str,
                        required=False,
                        default='/model/models/',
                        help='Path to where your ensemble output must be stored. Default is /model/models')

    parser.add_argument('-c',
                        '--ensemble_config',
                        type=str,
                        required=False,
                        default='./ensemble-config.json',
                        help='Path to where ensemble config .json')


    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    logging.info(f"Args: {args}")

    create_ensemble(workflow_path=args.nvt_workflow_path,
                    hugectr_model_path=args.hugectr_model_path,
                    ensemble_output_path=args.ensemble_output_path,
                    ensemble_config_file=args.ensemble_config
                    )
