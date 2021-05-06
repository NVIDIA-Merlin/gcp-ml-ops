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


PROJECT_ID=${1:-"dl-tme"}
GCLOUD_KEY=${2:-"gcloud_key.json"}

COPY_CONTAINER=gcr.io/$PROJECT_ID/google-nvidia-cloud-sdk:0.5.1
TRAIN_CONTAINER=gcr.io/$PROJECT_ID/merlin/merlin-training:0.5.1
MONITOR_COMPONENT=gcr.io/$PROJECT_ID/monitoring:0.5.1
VALIDATE_CONTAINER=gcr.io/$PROJECT_ID/validation:0.5.1

bash build_copy_container.sh $PROJECT_ID $GCLOUD_KEY
COPY_CONTAINER=$(docker inspect --format="{{index .RepoDigests 0}}" gcr.io/$PROJECT_ID/google-nvidia-cloud-sdk:0.5.1)
DEPLOY_CONTAINER=$COPY_CONTAINER

bash build_validation_component.sh $PROJECT_ID $GCLOUD_KEY
VALIDATE_CONTAINER=$(docker inspect --format="{{index .RepoDigests 0}}" gcr.io/$PROJECT_ID/validation:0.5.1)

bash build_training_container.sh $PROJECT_ID $GCLOUD_KEY
TRAIN_CONTAINER=$(docker inspect --format="{{index .RepoDigests 0}}" gcr.io/$PROJECT_ID/merlin/merlin-training:0.5.1)

bash build_monitoring_component.sh $PROJECT_ID $GCLOUD_KEY
MONITOR_COMPONENT=$(docker inspect --format="{{index .RepoDigests 0}}" gcr.io/$PROJECT_ID/monitoring:0.5.1)


source activate mlpipeline
python3 merlin-pipeline.py -vc $VALIDATE_CONTAINER -dex $COPY_CONTAINER -tc $TRAIN_CONTAINER -dc $DEPLOY_CONTAINER -mc $MONITOR_COMPONENT