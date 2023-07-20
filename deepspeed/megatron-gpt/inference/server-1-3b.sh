#!/bin/bash
# This example will start serving the 1.3B model.
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=output/gpt-1-3b/model
VOCAB_FILE=examples/deepspeed/megatron-gpt/tokenizer/wiki-en-tokenizer/vocab.json
MERGE_FILE=examples/deepspeed/megatron-gpt/tokenizer/wiki-en-tokenizer/merges.txt

export CUDA_DEVICE_MAX_CONNECTIONS=1

pip install pybind11 Ninja flask-restful

torchrun $DISTRIBUTED_ARGS Megatron-LM/tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 24  \
       --hidden-size 2048  \
       --load ${CHECKPOINT}  \
       --num-attention-heads 16  \
       --max-position-embeddings 2048  \
       --tokenizer-type GPT2BPETokenizer  \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 2048  \
       --out-seq-length 2048  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --top_p 0.9  \
       --seed 42
