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


import json, sys, argparse, os


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()    

    parser.add_argument("--model_version",
                        type=int,
                        required=True,
                        default=1,
                        help="Provide model version")

    parser.add_argument("--dcn_path",
                        type=str,
                        required=True,
                        default="/var/lib/data/script/dcn_files/dcn.json",
                        help="Path of original DCN")


    args = parser.parse_args()

    dcn = os.path.basename(args.dcn_path)
    dir_path = os.path.dirname(args.dcn_path)
    obj = None
    with open(args.dcn_path, "r") as f:
        obj = json.load(f)
    obj["inference"]["dense_model_file"] = "/model/models/dcn/" + str(args.model_version) + "/_dense_500.model"
    obj["inference"]["sparse_model_file"] = "/model/models/dcn/" + str(args.model_version) + "/0_sparse_500.model"
    # print(obj["inference"]["dense_model_file"])
    updated_json = dir_path+"/dcn" + str(args.model_version) + ".json"
    with open(updated_json,"w") as f:
        json.dump(obj, f)

    