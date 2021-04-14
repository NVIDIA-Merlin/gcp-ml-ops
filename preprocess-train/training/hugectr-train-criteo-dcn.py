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
import argparse
import logging

import hugectr
from mpi4py import MPI


def train(input_train, input_val, max_iter,
                batchsize, snapshot, num_gpus, eval_interval,
                dense_model_file, sparse_model_files):

    logging.info(f"GPU Devices: {num_gpus}")

    # Configure and define the HugeCTR model
    solver = hugectr.solver_parser_helper(num_epochs = 0,
                                        max_iter = max_iter,
                                        max_eval_batches = 100,
                                        batchsize_eval = batchsize,
                                        batchsize = batchsize,
                                        model_file = dense_model_file,
                                        embedding_files = sparse_model_files,
                                        display = 200,
                                        eval_interval = eval_interval,
                                        i64_input_key = True,
                                        use_mixed_precision = False,
                                        repeat_dataset = True,
                                        snapshot = snapshot,
                                        vvgpu = [num_gpus],
                                        use_cuda_graph = False
                                        )

    optimizer = hugectr.optimizer.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                        use_mixed_precision = False)
    model = hugectr.Model(solver, optimizer)

    # The slot_size_array are the cardinalities of each categorical feature after NVTabular preprocessing
    model.add(hugectr.Input(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                source = input_train,
                                eval_source = input_val,
                                check_type = hugectr.Check_t.Non,
                                label_dim = 1, label_name = "label",
                                dense_dim = 13, dense_name = "dense",
                                slot_size_array = [18576837, 29428, 15128, 7296, 19902, 4, 6466, 1311, 62, 11700067, 622921, 219557, 11, 2209, 9780, 71, 4, 964, 15, 22022124, 4384510, 15960286, 290588, 10830, 96, 35],
                                data_reader_sparse_param_array =
                                [hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Distributed, 30, 1, 26)],
                                sparse_names = ["data1"]))

    # Sparse Embedding Layer
    model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
                                max_vocabulary_size_per_gpu = 88656602,
                                embedding_vec_size = 16,
                                combiner = 0,
                                sparse_embedding_name = "sparse_embedding1",
                                bottom_name = "data1"))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                                bottom_names = ["sparse_embedding1"],
                                top_names = ["reshape1"],
                                leading_dim=416))

    # Concatenate sparse embedding and dense input
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                                bottom_names = ["reshape1", "dense"], top_names = ["concat1"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Slice,
                                bottom_names = ["concat1"],
                                top_names = ["slice11", "slice12"],
                                ranges=[(0,429),(0,429)]))

    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCross,
                                bottom_names = ["slice11"],
                                top_names = ["multicross1"],
                                num_layers=6))

    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["slice12"],
                                top_names = ["fc1"],
                                num_output=1024))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc1"],
                                top_names = ["relu1"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                                bottom_names = ["relu1"],
                                top_names = ["dropout1"],
                                dropout_rate=0.5))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["dropout1"],
                                top_names = ["fc2"],
                                num_output=1024))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc2"],
                                top_names = ["relu2"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                                bottom_names = ["relu2"],
                                top_names = ["dropout2"],
                                dropout_rate=0.5))

    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                                bottom_names = ["dropout2", "multicross1"],
                                top_names = ["concat2"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["concat2"],
                                top_names = ["fc3"],
                                num_output=1))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                                bottom_names = ["fc3", "label"],
                                top_names = ["loss"]))
    model.compile()
    model.summary()
    model.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--input_train',
                        type=str,
                        required=False,
                        default='/mlops/scripts/test-script/test_dask/output/train/_file_list.txt',
                        help='Path to training data _file_list.txt')

    parser.add_argument('-v',
                        '--input_val',
                        type=str,
                        required=False,
                        default='/mlops/scripts/test-script/test_dask/output/valid/_file_list.txt',
                        help='Path to validation data _file_list.txt')

    parser.add_argument('-i',
                        '--max_iter',
                        type=int,
                        required=False,
                        default=20000,
                        help='Number of training iterations')

    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=2048,
                        help='Batch size')

    parser.add_argument('-s',
                        '--snapshot',
                        type=int,
                        required=False,
                        default=10000,
                        help='Saves a model snapshot after given number of iterations')

    parser.add_argument('-g',
                        '--num_gpus',
                        nargs='+',
                        type=int,
                        required=False,
                        default=[0,1],
                        help='GPU devices to use for Preprocessing')

    parser.add_argument('-r',
                        '--eval_interval',
                        type=int,
                        required=False,
                        default=1000,
                        help='Run evaluation after given number of iterations')

    parser.add_argument('-d',
                        '--dense_model_file',
                        type=str,
                        required=False,
                        default=None,
                        help='Path to an existing dense model. If provided, resumes training from here. Eg. ./_dense_19500.model ')

    parser.add_argument('-m',
                        '--sparse_model_files',
                        type=str,
                        nargs='+',
                        required=False,
                        default=None,
                        help='Paths to an existing sparse snapshots. If provided, resumes training from here. Eg. --sparse_model_files ./model-snapshot/0_sparse_19500.model ./model-snapshot/0_sparse_19500.model')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    logging.info(f"Args: {args}")

    # Both the dense and sparse model files should be provided if either one is provided
    if args.dense_model_file and args.sparse_model_files:
        logging.info("Training from previously saved model...")
        logging.info(f"Dense model file: {args.dense_model_file}")
        logging.info(f"Sparse model file: {args.sparse_model_files}")
        dense_model_file = args.dense_model_file
        sparse_model_files = args.sparse_model_files
    elif (args.dense_model_file and args.sparse_model_files is None) or \
                            (args.sparse_model_files and args.dense_model_file is None):
        parser.error("--dense_model_file and --sparse_model_files both need to be provided together.")
    else:
        logging.info("No previous checkpoint/model provided. Training from scratch. ")
        dense_model_file = ""
        sparse_model_files = []

    train(input_train=args.input_train,
            input_val=args.input_val,
            max_iter=args.max_iter,
            batchsize=args.batchsize,
            snapshot=args.snapshot,
            eval_interval=args.eval_interval,
            num_gpus=args.num_gpus,
            dense_model_file=dense_model_file,
            sparse_model_files=sparse_model_files
            )