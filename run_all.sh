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

COPY_CONTAINER=gcr.io/$PROJECT_ID/google-nvidia-cloud-sdk:0.4
TRAIN_CONTAINER=gcr.io/$PROJECT_ID/merlin/merlin-training:0.4
MONITOR_COMPONENT=gcr.io/$PROJECT_ID/monitoring:0.4
VALIDATE_CONTAINER=gcr.io/$PROJECT_ID/validation:0.4

bash build_copy_container.sh
COPY_CONTAINER=$(docker inspect --format="{{index .RepoDigests 0}}" gcr.io/$PROJECT_ID/google-nvidia-cloud-sdk:0.4)
DEPLOY_CONTAINER=$COPY_CONTAINER

# bash build_validation_component.sh
# VALIDATE_CONTAINER=$(docker inspect --format="{{index .RepoDigests 0}}" gcr.io/$PROJECT_ID/validation:0.4)

bash build_training_container.sh
TRAIN_CONTAINER=$(docker inspect --format="{{index .RepoDigests 0}}" gcr.io/$PROJECT_ID/merlin/merlin-training:0.4)

bash build_monitoring_component.sh
MONITOR_COMPONENT=$(docker inspect --format="{{index .RepoDigests 0}}" gcr.io/$PROJECT_ID/monitoring:0.4)


source activate mlpipeline
# python3 merlin-pipeline.py -vc $VALIDATE_CONTAINER -cc $COPY_CONTAINER -tc $TRAIN_CONTAINER -dc $DEPLOY_CONTAINER -mc $MONITOR_COMPONENT