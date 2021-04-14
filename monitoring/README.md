# Monitoring

## Pre-requisites
- Dependencies to run this test are available in the monitoring container, including pre-set env vars. eg.
```
docker run -it --rm --net=host -v ${PWD}/:/workdir gcr.io/dl-tme/monitoring-container:0.1
```
- Triton needs to be up and running, hosting the ensemble model

## Monitoring Agent
```
python3 perf-monitor.py --project_id $PROJECT --subscription_id sub_one --evaluate_period 200 --min_trigger_len 0.5 --acc_threshold 0.8
```


## Inference Client to Test Monitoring
```
python3 perf-monitor-test.py  --triton_grpc_url localhost:8001 --model_name dcn_ens --verbose False -d /workdir/data/day_23.parquet --batch_size 64 --n_batches=5 --project_id $PROJECT --topic_id hello_pubsub
```
