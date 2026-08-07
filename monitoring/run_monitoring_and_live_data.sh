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
GCS_BUCKET=${2:-"criteo-data"}
BUCKET_PATH=${3:-"new_data"}
LOCAL=${4:-"/var/lib/data/new_data"}
PIPELINE=${5:-"merlin-pipeline"}
PUBSUB=${6:-"mlops-test-sub"}
PUBSUB_TOPIC=${7:-"mlops-test"}
TRIGGER_COOLDOWN_SECONDS=${8:-3600}
EXPECTED_MODEL_NAME=${9:-"dcn_ens"}
KFP_HOST=${KFP_HOST:?"KFP_HOST must be configured"}


echo "perf monitor"
python3 -u /script/perf-monitor.py --PV_loc "$LOCAL" --project_id "$PROJECT_ID" --subscription_id "$PUBSUB" --topic_id "$PUBSUB_TOPIC" --expected_model_name "$EXPECTED_MODEL_NAME" --evaluate_period 200 --min_trigger_len 0.5 --acc_threshold 0.8 --trigger_cooldown_seconds "$TRIGGER_COOLDOWN_SECONDS" --pipeline_name "$PIPELINE" --kfp_host "$KFP_HOST" &

echo "gcs"
python3 -u /script/csv_read_gcs_write.py --pv_dir $LOCAL  --sleep_time 10 --bucket $GCS_BUCKET --bucket_path $BUCKET_PATH

echo "done"