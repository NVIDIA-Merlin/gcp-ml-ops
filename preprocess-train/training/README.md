## Training

If training from scratch:
```
python3 hugectr-train-criteo-dcn.py --input_train ./test_dask/output/train/_file_list.txt --input_val ./test_dask/output_val/train/_file_list.txt --max_iter 2000 --snapshot 500 --num_gpus 0 1 2 3 --eval_interval 1000
```
If continuing training from a checkpoint. (You need to provide both dense model file and sparse model file(s))
```
python3 hugectr-train-criteo-dcn.py --input_train ./test_dask/output/train/_file_list.txt --input_val ./test_dask/output_val/train/_file_list.txt --max_iter 2000 --snapshot 500 --num_gpus 0 1 2 3 --eval_interval 1000 --dense_model_file ./model-snapshot/_dense_19500.model --sparse_model_files ./model-snapshot/0_sparse_19500.model
```

## Create model directoy
- Create a model directory for the trained model: `mkdir -p /model/criteo_hugectr/1`
- Move the model files:  `mv *.model /model/movielens_hugectr/1/`

## Create inference configuration .json
- Create the inference dcn.json at `/model/criteo_hugectr/1/dcn.json`. This is the inference .json file with the correct sparse and dense model file paths (relative to how it will appear from the Merlin Inference container).

## Generate Model Ensemble
```
python3 create-nvt-hugectr-ensemble.py --nvt_workflow_path ./test_dask/output/workflow/ --hugectr_model_path /model/criteo_hugectr/1/ --ensemble_output_path /model/models/ --ensemble_config ./ensemble-config.json
```
This will create the model ensemble at /model/models. This should be mounted to the Merlin Inference container to host the ensemble model. The latest model version is extraced from the hugectr_model_path str.
