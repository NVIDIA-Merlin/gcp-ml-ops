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

import pandas as pd
import numpy as np
from pyarrow import csv, parquet
from glob import glob
import os
from datetime import datetime
import re
import sys
from google.cloud import storage
import argparse
from time import sleep

PATH = 'dummy'

def get_local_files(path):
    local_files = glob(path+"/*")
    return local_files

def files_to_data_frames(local_files):
    data_frames = []
    for local_file in local_files:
        df = pd.read_csv(local_file)
        data_frames.append(df)
        del df
    return data_frames

def files_to_data_frames_parquet(local_files):
    data_frames = []
    for local_file in local_files:
        df = pd.read_parquet(local_file, engine='pyarrow')
        data_frames.append(df)
        del df
    return data_frames

def one_giant_data_frame(data_frames):
    big_un = pd.concat(data_frames, copy=False)
    return big_un

def file_to_data_frame_to_parquet(data_frame, parquet_file):
    # table = csv.read_csv(local_file)
    # parquet.write_table(table, parquet_file)
    data_frame.to_parquet(parquet_file, engine='pyarrow')

class GCSStore:
    def __init__(self, bucket_name, bucket_path):
        self.bucket_name = bucket_name
        self.bucket_path = bucket_path
        # Create a Cloud Storage client.
        self.gcs = storage.Client()

        # Get the bucket that the file will be uploaded to.
        self.bucket = self.gcs.get_bucket(self.bucket_name)


    def list_bucket(self, limit=sys.maxsize):
        a_bucket = self.gcs.lookup_bucket(self.bucket_name)
        bucket_iterator = a_bucket.list_blobs(prefix=self.bucket_path)
        for resource in bucket_iterator:
            print(resource.name)
            limit = limit - 1
            if limit <= 0:
                break

    def upload_to_bucket(self, input_file_name, output_file_name):
        blob2 = self.bucket.blob(self.bucket_path + "/" + output_file_name)
        blob2.upload_from_filename(filename=input_file_name)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    print("In read-write csv to parquet")

    parser.add_argument("--pv_dir",
                        type=str,
                        required=True,
                        default="/var/lib/data/new_data",
                        help="Path to new data in PV")

    parser.add_argument("--sleep_time",
                        type=int,
                        required=True,
                        default=1,
                        help="Sleep time in seconds")

    parser.add_argument("--bucket",
                        type=str,
                        required=True,
                        default="criteo-data",
                        help="Name of GCS bucket")

    parser.add_argument("--bucket_path",
                        type=str,
                        required=True,
                        default="new_data",
                        help="Path of directory to store files on GCS bucket")
    
    args = parser.parse_args()

    sleep_time = args.sleep_time
    gcs_store = GCSStore(args.bucket, args.bucket_path)

    while True:
        sleep(sleep_time)
        local_files = get_local_files(args.pv_dir)
        if len(local_files) == 0:
            print("No files to process. Sleeping for {} secs".format(sleep_time))
            continue
        
        print("New files found. Pushing to GCS...")
        for each_file in local_files:
            print("pushing {} to {}".format(each_file, args.bucket + "/" + args.bucket_path + "/" +os.path.basename(each_file)))
            gcs_store.upload_to_bucket(each_file, os.path.basename(each_file))
            print("Uploaded {} to {} at {}. Deleting {} from PV".format(each_file,
                                    args.bucket + "/" + args.bucket_path + "/" +os.path.basename(each_file),
                                    datetime.now(), each_file))
            os.remove(each_file)
        

