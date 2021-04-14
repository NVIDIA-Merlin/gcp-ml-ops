## Run the Merlin Training container
```
docker run --gpus=all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE  -v ${PWD}:/mlops  -v /data/crit_int_pq:/crit_int_pq <Merlin-Training-Container> /bin/bash
```

## Activate conda environment
```
source activate rapids
```

## Full Preprocessing
```
python3 nvt-preprocess.py -d /crit_int_pq -o ./ -t 1 -v 1 -g 0 1
```


## Incremental Preprocessing
```
python3 nvt-preprocess-incremental.py --input_train_dir ./train-data/ --output_dir ./test_dask/output --workflow_dir ./test_dask/output/workflow/ --dask_workdir ./test_dask/workdir --num_gpus 0 1
```


## Incremental Preprocessing
```
python3 nvt-preprocess-incremental.py --input_train_dir ./train-data/ --output_dir ./test_dask/output --workflow_dir ./test_dask/output/workflow/ --dask_workdir ./test_dask/workdir --num_gpus 0 1
```
