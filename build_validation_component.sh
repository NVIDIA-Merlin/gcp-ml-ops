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


PROJECT_ID=${1:-"dl-tme"} # Google Cloud project ID
GCLOUD_KEY=${2:-"gcloud_key.json"} # Path to Google Cloud key

image_name=gcr.io/$PROJECT_ID/validation # Specify the image name here
image_tag=0.5.1

full_image_name=${image_name}:${image_tag}

# docker build --build-arg gcloud_key=$GCLOUD_KEY --build-arg project_id=$PROJECT_ID --no-cache -f Dockerfile.validation -t $full_image_name  .
docker build --build-arg gcloud_key=$GCLOUD_KEY --build-arg project_id=$PROJECT_ID -f Dockerfile.validation -t $full_image_name  .

printf "\n\nPushing the container on GCR..."
docker push $full_image_name

printf "\n\n\n\n<<< Unique ID of the container is below. Use this ID in the pipeline component >>>\n\n"
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"
printf "\n\n------------------------------------------------------------------------------------------\n\n"