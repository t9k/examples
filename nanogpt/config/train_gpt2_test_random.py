# config for test of training GPT-2 (124M) on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~20-30 min:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_test.py

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5

# this makes total number of tokens be 1B
max_iters = 2000
lr_decay_iters = 2000

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# disable wandb
wandb_log = False

# PyTorch 2.0 can compile
compile = True

random_data = True
out_dir = 'out-test'
always_save_checkpoint = False
