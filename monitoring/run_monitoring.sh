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

PROJECT=${1:-"dl-tme"}
DOCKER_IMG=${2:-"gcr.io/${PROJECT}/monitoring:0.4"}
PIPELINE=${3:-"merlin-pipeline"}
GCS_BUCKET=${4:-"criteo-data"}
BUCKET_PATH=${5:-"new_data"}
LOCAL=${6:-"/var/lib/data"}
PUBSUB=${7:-"mlops-test-sub"}

gcloud auth activate-service-account --key-file=/script/gcloud_key.json
gcloud container clusters get-credentials cluster-4 --zone us-central1-a --project $PROJECT

monitoring_status=$(helm status monitoring 2>&1)
echo "monitoring status: "
echo $monitoring_status
if [[ "$monitoring_status" == "Error: release: not found" ]]; then
    helm install monitoring --set project_id=$PROJECT --set image.repository=$DOCKER_IMG --set pipeline=$PIPELINE --set gcs_bucket=$GCS_BUCKET --set bucket_path=$BUCKET_PATH --set local=$LOCAL --set pubsub=$PUBSUB /script
else
    echo "Monitoring module running already, not deploying another instance"
fi