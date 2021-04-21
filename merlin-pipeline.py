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


import os
import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.components as comp
import kfp.dsl as dsl
import datetime
import os
from kubernetes import client as k8s_client
import argparse
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

args = None
accelerator = 'nvidia-tesla-a100'
node_pool = 'a100-pool'
high_mem_node = 'high-mem-node'
gcs_bucket_head = ''

@dsl.pipeline(
    name="Merlin pipeline",
    description="HugeCTR training to deployment"
)
def merlin_pipeline(
  data_dir: 'GCSPath' = 'gs://criteo-data/dummy_data/*',
  gcs_bucket_head: str = 'criteo-data',
  local_data_dir: str = '/var/lib/data',
  project_id: str = 'dl-tme',
  pipeline_name: str = 'merlin-pipeline',
  new_data_collection: str = 'new_data'):
    
    global args, accelerator

    # Persistent volume variables
    persistent_volume_name = 'my-file-server'
    persistent_volume_claim_name = 'my-volume-claim'
    persistent_volume_path = '/var/lib/data'

    # First component - Copy data from GCS to PV
    copy_data = dsl.ContainerOp(
      name="data-extraction",
      image=args.data_extraction,
      command=["bash" , "/script/run_copy_merlin.sh"],
      arguments=[data_dir, local_data_dir, project_id]
    )

    # Second component - Data validation
    data_validation = dsl.ContainerOp(
      name="validate-data",
      image=args.validate_container,
      command=["bash" , "/script/run_validation.sh"],
      arguments=[local_data_dir]
    )

    # Third component - Preprocess and Train
    preprocess_train = dsl.ContainerOp(
      name="merlin-preprocess-train",
      image=args.preprocess_train_container,
      command=["bash", "/script/preprocess-train.sh"],
      arguments=[local_data_dir, project_id]
    )

    # Fourth component - Model deployment
    deploy_triton = dsl.ContainerOp(
      name="triton-inference",
      image=args.deploy_container,
      command=["bash" , "/script/run_merlin_inference.sh"],
      arguments=[local_data_dir, project_id, "/script/gcloud_key.json"]
    )

    # Fifth component - Monitoring
    monitoring = dsl.ContainerOp(
      name="data-monitoring",
      image=args.monitor_container,
      command=["bash" , "/script/run_monitoring.sh"],
      arguments=[project_id, args.monitor_container, pipeline_name, gcs_bucket_head, new_data_collection, local_data_dir]
    )


    # Adding PV, PVC, GPU constraints to the components
    copy_data.add_volume(k8s_client.V1Volume(name=persistent_volume_name,
      persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
      claim_name=persistent_volume_claim_name))).add_volume_mount(k8s_client.V1VolumeMount(
      mount_path=persistent_volume_path,name=persistent_volume_name)).set_gpu_limit(1).add_node_selector_constraint('cloud.google.com/gke-accelerator', accelerator).add_node_selector_constraint('cloud.google.com/gke-nodepool', node_pool)

    data_validation.add_volume(k8s_client.V1Volume(name=persistent_volume_name,
      persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
      claim_name=persistent_volume_claim_name))).add_volume_mount(k8s_client.V1VolumeMount(
      mount_path=persistent_volume_path,name=persistent_volume_name)).add_node_selector_constraint('cloud.google.com/gke-nodepool', high_mem_node)

    preprocess_train.add_volume(k8s_client.V1Volume(name=persistent_volume_name,
      persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
      claim_name=persistent_volume_claim_name))).add_volume_mount(k8s_client.V1VolumeMount(
      mount_path=persistent_volume_path,name=persistent_volume_name)).set_gpu_limit(1).add_node_selector_constraint('cloud.google.com/gke-accelerator', accelerator).add_node_selector_constraint('cloud.google.com/gke-nodepool', node_pool)

    deploy_triton.add_volume(k8s_client.V1Volume(name=persistent_volume_name,
      persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
      claim_name=persistent_volume_claim_name))).add_volume_mount(k8s_client.V1VolumeMount(
      mount_path=persistent_volume_path,name=persistent_volume_name)).set_gpu_limit(1)

    # Sequencing the components
    data_validation.after(copy_data)
    preprocess_train.after(data_validation)
    deploy_triton.after(preprocess_train)
    monitoring.after(deploy_triton)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Parse command line arguments
  parser.add_argument("-vc",
                      "--validate_container",
                        type=str,
                        required=False,
                        help="pass validate data container")

  parser.add_argument("-dex",
                      "--data_extraction",
                        type=str,
                        required=True,
                        help="pass copy container")

  parser.add_argument("-tc",
                      "--preprocess_train_container",
                        type=str,
                        required=True,
                        help="pass preprocess-train container")
  
  parser.add_argument("-dc",
                      "--deploy_container",
                        type=str,
                        required=True,
                        help="pass copy container")

  parser.add_argument("-mc",
                      "--monitor_container",
                        type=str,
                        required=True,
                        help="pass copy container")

  args = parser.parse_args()

  logger.info("Data extraction container: " + args.data_extraction)
  logger.info("Validate container: " + args.validate_container)
  logger.info("Preprocess-train container: " + args.preprocess_train_container)
  logger.info("Deploy container: " + args.deploy_container)
  logger.info("Monitor container: " + args.monitor_container)


  import kfp.compiler as compiler
  # Export pipeline as .tar.gz
  compiler.Compiler().compile(merlin_pipeline, __file__ + '.tar.gz')
  
