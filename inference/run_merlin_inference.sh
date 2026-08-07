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

PV_LOC=${1:-"/var/lib/data"}
PROJECT_ID=${2:-"dl-tme"}
CLUSTER=${3:-"merlin-mlops"}
ZONE=${4:-"us-central1-a"}
WORKLOAD_SERVICE_ACCOUNT=${5:-"mlops-workload"}
TRITON_IMAGE=${6:?"Triton image digest must be provided"}

gcloud container clusters get-credentials $CLUSTER --zone $ZONE --project $PROJECT_ID
gcloud config set project $PROJECT_ID

if ! [ -d $PV_LOC/inference ]; then
    mkdir $PV_LOC/inference
fi

triton_status=$(helm status triton 2>&1)
echo "Triton status: "
echo $triton_status
if [[ "$triton_status" == "Error: release: not found" ]]; then
    cp /script/load-triton-ensemble.py $PV_LOC/inference/load-triton-ensemble.py
    cp /script/triton/run_triton.sh $PV_LOC/inference/run_triton.sh

    helm install triton /script/triton/ --set image.imageName="$TRITON_IMAGE" --set serviceAccountName="$WORKLOAD_SERVICE_ACCOUNT"
else
    echo "Triton running already, not deploying another instance."
fi
