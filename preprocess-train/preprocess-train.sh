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
set -e

PV_LOC=${1:-"/var/lib/data"}
PROJECT=${2:-"dl-tme"}
cluster=${3:-"merlin-mlops"}
zone=${4:-"us-central1-a"}

cp -r /script $PV_LOC

#echo "Preprocessing..."
cd $PV_LOC
echo $PV_LOC

gcloud auth activate-service-account --key-file=/script/gcloud_key.json
gcloud container clusters get-credentials $cluster --zone $zone --project $PROJECT
gcloud config set project $PROJECT

# Check if triton is deployed
triton_status=$(helm status triton 2>&1)
echo "Triton status: "
echo $triton_status
if [[ "$triton_status" == "Error: release: not found" ]]; then
    echo "Triton is not running. This is first deployment."
    echo "Preprocessing...."
    ls -al $PV_LOC/criteo-data/crit_int_pq
    python3 -u $PV_LOC/script/preprocessing/nvt-preprocess.py -d $PV_LOC/criteo-data/crit_int_pq -o $PV_LOC/criteo-data/ -t 1 -v 1 -g 0

    echo "Training..."
    python3 -u $PV_LOC/script/training/hugectr-train-criteo-dcn.py --input_train $PV_LOC/criteo-data/test_dask/output/train/_file_list.txt --input_val $PV_LOC/criteo-data/test_dask/output/valid/_file_list.txt --max_iter 600 --snapshot 500 --num_gpus 0

    mkdir -p $PV_LOC/model/criteo_hugectr/1/
    mv $PV_LOC/*.model $PV_LOC/model/criteo_hugectr/1/

    mkdir -p $PV_LOC/models/

    echo "Create ensemble"
    python3 -u $PV_LOC/script/training/create-nvt-hugectr-ensemble.py --nvt_workflow_path $PV_LOC/criteo-data/test_dask/output/workflow/ --hugectr_model_path $PV_LOC/model/criteo_hugectr/1/ --ensemble_output_path $PV_LOC/models/  --ensemble_config $PV_LOC/script/training/ensemble-config.json

    echo "Copy dcn.json"
    cp $PV_LOC/script/dcn_files/dcn.json $PV_LOC/models/dcn/1

else
    echo "Triton is running. This is triggered run. Running incremental pre-processing"
    echo "Incremental preprocessing..."
    ls -al $PV_LOC/criteo-data/new_data
    python3 -u $PV_LOC/script/preprocessing/nvt-preprocess-incremental.py --input_train_dir $PV_LOC/criteo-data/new_data/ --output_dir $PV_LOC/criteo-data/output --workflow_dir $PV_LOC/criteo-data/test_dask/output/workflow/ --dask_workdir $PV_LOC/criteo-data/test_dask/workdir --num_gpus 0 

    previous_version=$(ls $PV_LOC/model/criteo_hugectr/ -v | tail -n1)

    echo "Incremental Training..."
    python3 -u $PV_LOC/script/training/hugectr-train-criteo-dcn.py --input_train $PV_LOC/criteo-data/test_dask/output/train/_file_list.txt --input_val $PV_LOC/criteo-data/test_dask/output/valid/_file_list.txt --max_iter 600 --snapshot 500 --num_gpus 0 --dense_model_file $PV_LOC/model/criteo_hugectr/$previous_version/_dense_500.model --sparse_model_files $PV_LOC/model/criteo_hugectr/$previous_version/0_sparse_500.model

    new_version="$(($previous_version + 1))" 

    mkdir -p $PV_LOC/model/criteo_hugectr/$new_version/

    mv $PV_LOC/*.model $PV_LOC/model/criteo_hugectr/$new_version/

    mkdir -p $PV_LOC/models_recurrent_runs

    echo "Incremental Create ensemble"
    python3 -u $PV_LOC/script/training/create-nvt-hugectr-ensemble.py --nvt_workflow_path $PV_LOC/criteo-data/test_dask/output/workflow/ --hugectr_model_path $PV_LOC/model/criteo_hugectr/$new_version/ --ensemble_output_path $PV_LOC/models_recurrent_runs --ensemble_config $PV_LOC/script/training/ensemble-config.json

    python3 -u $PV_LOC/script/dcn_files/format_dcn.py --model_version $new_version --dcn_path $PV_LOC/script/dcn_files/dcn.json

    mv $PV_LOC/models_recurrent_runs/dcn/1 $PV_LOC/models/dcn/$new_version
    mv $PV_LOC/models_recurrent_runs/dcn/config.pbtxt $PV_LOC/models/dcn/
    cp $PV_LOC/script/dcn_files/dcn$new_version.json $PV_LOC/models/dcn/$new_version/dcn.json

    mv $PV_LOC/models_recurrent_runs/dcn_ens/1 $PV_LOC/models/dcn_ens/$new_version
    mv $PV_LOC/models_recurrent_runs/dcn_nvt/1 $PV_LOC/models/dcn_nvt/$new_version

    rm -rf $PV_LOC/models_recurrent_runs
fi  
