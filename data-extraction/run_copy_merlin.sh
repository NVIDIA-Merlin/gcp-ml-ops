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

data_input_path=$1;
data_local=$2;
project_id=$3;
new_data_path=$4;
cluster=$5;
zone=$6;

gcloud auth activate-service-account --key-file=/script/gcloud_key.json
gcloud container clusters get-credentials $cluster --zone $zone --project $project_id
gcloud config set project $project_id

triton_status=$(helm status triton 2>&1)
if [[ "$triton_status" == "Error: release: not found" ]]; then
  if [ -d "$data_local" ]; then
    ### Take action if $DIR exists ###
    echo "Running first time..."
    echo "Directory ${DIR} exists. Copying files from gcs"
    gsutil list gs://
    if ! [ -d "$data_local/criteo-data" ]; then
      echo "Making criteo-data"
      mkdir -p $data_local/criteo-data/crit_int_pq
    fi
    echo "Copying data..."
    gsutil cp -r $data_input_path $data_local/criteo-data/crit_int_pq
    echo "Copying done"
    
    for entry in "$data_local/criteo-data/crit_int_pq"/*
    do
      echo "$entry"
    done

  else
    ###  Control will jump here if $DIR does NOT exists ###
    echo "Error: ${DIR} not found. Can not continue."
    exit 1
  fi
  echo "copying done"
else
  if [ -d "$data_local" ]; then
    ### Take action if $DIR exists ###
    echo "Recurrent run..."
    echo "Directory ${DIR} exists. Copying files from gcs"
    # gsutil list gs://
    if ! [ -d "$data_local/criteo-data/new_data" ]; then
      echo "Making criteo-data"
      mkdir -p $data_local/criteo-data/new_data
    fi
    echo "Copying data..."
    gsutil cp -r $new_data_path $data_local/criteo-data/new_data
    echo "Copying done"
    
    for entry in "$data_local/criteo-data/new_data"/*
    do
      echo "$entry"
    done
  else
    ###  Control will jump here if $DIR does NOT exists ###
    echo "Error: ${DIR} not found. Can not continue."
    exit 1
  fi
fi

