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


echo "perf monitor"
python3 -u /script/perf-monitor.py --PV_loc /var/lib/data/ --project_id dl-tme --subscription_id sub_one --evaluate_period 200 --min_trigger_len 0.5 --acc_threshold 0.8 --pipeline_name merlin-pipeline &

echo "gcs"
python3 -u /script/csv_read_gcs_write.py --pv_dir /var/lib/data/  --sleep_time 10 --bucket criteo-data --bucket_path new_data

echo "done"