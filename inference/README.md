##  Host the model for inference

1. Ensure that you generated the model ensemble successfully. The folder should look something like this

```
cd /model
tree
.
-- models
    |-- dcn
    |   |-- 1
    |   |   |-- 0_sparse_1500.model
    |   |   |-- _dense_1500.model
    |   |   `-- dcn.json
    |   `-- config.pbtxt
    |-- dcn_ens
    |   |-- 1
    |   `-- config.pbtxt
    `-- dcn_nvt
        |-- 1
        |   |-- model.py
        |   `-- workflow
        |       |-- categories
        |       |   |-- unique.C1.parquet
        |       |   |-- unique.C10.parquet
        |       |   |-- unique.C11.parquet
        |       |   |-- unique.C12.parquet
		.
		.
```

3. Run the Merlin Inference Container. Make sure to mount the models directory to /model.
```
docker run --gpus=all -it -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model:/model -v {PWD}/workdir:/workdir nvcr.io/nvidia/merlin/merlin-inference:0.4
```

4. Activate rapids environment
```
source activate rapids
```

5. Run the Triton server
```
tritonserver --model-repository=/model/models/  --backend-config=hugectr,dcn=/model/models/dcn/1/dcn.json --backend-config=hugectr,supportlonglong=true --model-control-mode=explicit 
```

Note that this will run Triton, but you will need to load the ensemble model into Triton


## Load the model ensemble
From another terminal into the same inference container, (or a different execution instance of the merlin inference container) which has access to the raw dataset for testing purposes.
```
python3 /workdir/load-triton-ensemble.py --triton_grpc_url localhost:8001 --model_name dcn_ens --verbose False
```

## Run the client app
```
python3 /workdir/inference-client.py  --triton_grpc_url localhost:8001 --model_name dcn_ens --verbose False -d /crit_int_pq/day_23.parquet --batch_size 64 --n_batches=5
```









1. Add your dense and sparse .model files to ./models/dcn/1.

2. Update ./models/dcn/dcn.json file to reflect your model paths. These paths should be according to how it will be visible from inside the docker container after you mount the /model directory
```
    "dense_model_file": "/model/dcn/1/_dense_10000.model",
    "sparse_model_file": "/model/dcn/1/0_sparse_10000.model",
```

3. Run the Merlin Inference Container. Make sure to mount the models directory to /model.
```
docker run --gpus=all -it -p 8005:8000 -p 8004:8001 -p 8003:8002 -v ${PWD}/models:/model <merlin-inference-container> 
```

4. Run the Triton Inference server
```
 tritonserver --model-repository=/model/ --model-repository=/model/ --load-model=dcn \
	--model-control-mode=explicit \
	--backend-directory=/usr/local/hugectr/backends/ \
	--backend-config=hugectr,dcn=/model/dcn/1/dcn.json \
	--backend-config=hugectr,supportlonglong=true
```


## Criteo Inference Client app

1. You must have preprocessed test data available. This preprocessing should use the same workflow object as obtained during preprocessing of the training data


2. Start the Triton Client container. Remember to mount the criteo-inference-client.py app and directory where test data is stored
```
docker run --net=host  --gpus=all -it -v ${PWD}:/working_dir/ nvcr.io/nvidia/tritonserver:20.10-py3-clientsdk
```

3. Install Dependencies
```
apt update
apt install llvm-10*
pip3 install -y pandas
pip3 install --upgrade cython
pip3 install -y pyarrow==0.15.1
pip3 install sklearn
```

4. Run the criteo client app inside the container
```
python3 criteo-inference-client.py --test_dir ./test_dask/output/test/ --batch_size 64 --n_batches 2 --model dcn
```
