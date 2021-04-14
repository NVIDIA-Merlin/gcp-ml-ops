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

if [ -d "$PV_LOC/stats" ] && [ -d "$PV_LOC/stats/stats.txt" ]; then
    previous_version=$(ls $PV_LOC/stats/ -v | tail -n1)
    new_version="$(($previous_version + 1))" 
    new_file="$(ls $PV_LOC/criteo-data/new_data/ | shuf -n 1)"
    
    echo "Generating stats for training data..."
    python3 -u /script/generate-stats.py --data_dir $PV_LOC/criteo-data/new_data/$new_file --output_dir $PV_LOC/stats/ --file_name "stats"$new_version".txt" 

    echo "Validate stats..."
    python3 -u /script/validate-stats.py --stats_file_1 $PV_LOC/stats/stats.txt --stats_file_2  $PV_LOC/stats/"stats"$new_version".txt"

else
    mkdir -p $PV_LOC/stats/

    echo "Generating stats for training data..."
    python3 -u /script/generate-stats.py --data_dir $PV_LOC/criteo-data/crit_int_pq/day_0.parquet --output_dir $PV_LOC/stats/ --file_name "stats.txt"
fi
