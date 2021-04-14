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



# Standard Libraries
import os
from time import time
import re
import shutil
import glob
import warnings
import sys
import argparse
import logging

# External Dependencies
import numpy as np
import pandas as pd
import cupy as cp
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.utils import parse_bytes
from dask.delayed import delayed
import rmm

import nvtabular as nvt
from nvtabular.utils import _pynvml_mem_size, device_mem_size

def run_preprocessing(input_train_path, workflow_path, output_path, dask_workdir, num_gpus):
    fname = '{}.parquet'
    train_files = [i for i in os.listdir(input_train_path) if re.match(fname.format('.*'), i) is not None]
    train_paths = [os.path.join(input_train_path, filename) for filename in train_files]

    # Deploy a Dask Distributed Cluster
    # Single-Machine Multi-GPU Cluster
    protocol = "tcp"             # "tcp" or "ucx"
    visible_devices = ",".join([str(n) for n in num_gpus])  # Delect devices to place workers
    device_limit_frac = 0.4      # Spill GPU-Worker memory to host at this limit.
    device_pool_frac = 0.5
    part_mem_frac = 0.05

    # Use total device size to calculate args.device_limit_frac
    device_size = device_mem_size(kind="total")
    part_size = int(part_mem_frac * device_size)
    logging.info(f"Partition size: {part_size}")

    # Deploy Dask Distributed cluster only if asked for multiple GPUs
    if len(num_gpus) > 1:
        logging.info("Deploy Dask Distributed cluster...")

        device_limit = int(device_limit_frac * device_size)
        device_pool_size = int(device_pool_frac * device_size)

        logging.info("Checking if any device memory is already occupied...")
        # Check if any device memory is already occupied
        for dev in visible_devices.split(","):
            fmem = _pynvml_mem_size(kind="free", index=int(dev))
            used = (device_size - fmem) / 1e9
            if used > 1.0:
                warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

        cluster = None               # (Optional) Specify existing scheduler port
        if cluster is None:
            cluster = LocalCUDACluster(
                protocol = protocol,
                n_workers=len(visible_devices.split(",")),
                CUDA_VISIBLE_DEVICES = visible_devices,
                device_memory_limit = device_limit,
                local_directory=dask_workdir
            )

        logging.info("Create the distributed client...")
        # Create the distributed client
        client = Client(cluster)

        logging.info("Initialize memory pools...")
        # Initialize RMM pool on ALL workers
        def _rmm_pool():
            rmm.reinitialize(
                # RMM may require the pool size to be a multiple of 256.
                pool_allocator=True,
                initial_pool_size=(device_pool_size // 256) * 256, # Use default size
            )

        client.run(_rmm_pool)


    # Import the test .parquet
    logging.info("Importing Data...")
    test_dataset = nvt.Dataset(train_paths, engine='parquet', part_size=part_size)

    logging.info("Loading workflow object...")
    workflow = nvt.Workflow.load(workflow_path)

    # Specify the columns IDs: this part should exactly the columns while preproc. train, valid datasets
    CONTINUOUS_COLUMNS = ['I' + str(x) for x in range(1,14)]
    CATEGORICAL_COLUMNS =  ['C' + str(x) for x in range(1,27)]
    LABEL_COLUMNS = ['label']
    dict_dtypes={}

    for col in CATEGORICAL_COLUMNS:
        dict_dtypes[col] = np.int64

    for col in CONTINUOUS_COLUMNS:
        dict_dtypes[col] = np.float32

    for col in LABEL_COLUMNS:
        dict_dtypes[col] = np.float32

    # Create output directory for test data
    output_test_dir = os.path.join(output_path, 'train/')

    if not os.path.exists(output_test_dir):
        logging.info(f"Creating train/ directory at: {output_test_dir}")
        os.makedirs(output_test_dir)

    logging.info("Preprocessing Data...")
    workflow.transform(test_dataset).to_parquet(output_path=output_test_dir,
                                             dtypes=dict_dtypes,
                                             cats=CATEGORICAL_COLUMNS,
                                             conts=CONTINUOUS_COLUMNS,
                                             labels=LABEL_COLUMNS)

    logging.info("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--input_train_dir',
                        type=str,
                        required=False,
                        default='/crit_int_pq',
                        help='Path to Preprocessed Data Dir. Default is /crit_int_pq')

    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=False,
                        default='./test_dask/output/',
                        help='Path for Output directory. Default is ./test_dask/output/')

    parser.add_argument('-w',
                        '--workflow_dir',
                        type=str,
                        required=False,
                        default='./test_dask/output/workflow/',
                        help='Path to Saved Workflow object. This should be obtained from Preprocessing Training data. Default is ./test_dask/output/workflow')

    parser.add_argument('-e',
                        '--dask_workdir',
                        type=str,
                        required=False,
                        default='./test_dask/workdir',
                        help='Working directory for Dask. Default is ./test_dask/workdir')

    parser.add_argument('-g',
                        '--num_gpus',
                        nargs='+',
                        type=int,
                        required=False,
                        default=[0,1],
                        help='GPU devices to use for Preprocessing')

    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    logging.info(f"Args: {args}")

    run_preprocessing(input_train_path=args.input_train_dir,
                        workflow_path=args.workflow_dir,
                        output_path=args.output_dir ,
                        dask_workdir=args.dask_workdir,
                        num_gpus=args.num_gpus)
