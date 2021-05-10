#!/bin/bash

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

MODELS_DIR=${1:-"/model/models"}

set -m

source activate merlin

tritonserver --model-repository=$MODELS_DIR  --backend-config=hugectr,dcn=$MODELS_DIR/dcn/1/dcn.json --backend-config=hugectr,supportlonglong=true --model-control-mode=poll --repository-poll-secs=10 &

sleep 120

echo "starting script"
python3 /model/inference/load-triton-ensemble.py --triton_grpc_url localhost:8001 --model_name dcn_ens --verbose False

fg %1
