#!/bin/bash

########################################################
# Replication script for experiments with least squares.
########################################################

# Default training parameters
default_train_steps=1000000
default_scheduler_steprate=10000
default_learning_rate=1e-4

# Transformer and BaseConv. Sweep number of layers.
seq_ops=(attn base_conv)
n_layers=(1 2 4 8 12 16 24 32 48 64)
for seq_op in "${seq_ops[@]}"; do
    for n_layer in "${n_layers[@]}"; do
        WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/least_squares/least_squares.yaml --autoname False --model.n_layer ${n_layer} --model.seq_op ${seq_op} --training.train_steps ${default_train_steps} --training.scheduler_steprate ${default_scheduler_steprate} --training.learning_rate ${default_learning_rate} --wandb.name task=ls_seqop=${seq_op}_nlayer=${n_layer}
    done
done

# Train with fixed condition number
conds=(5)
for cond in "${conds[@]}"; do
    WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/least_squares/least_squares_cond=${cond}.yaml --autoname False --training.train_steps ${default_train_steps} --training.scheduler_steprate ${default_scheduler_steprate} --training.learning_rate ${default_learning_rate} --wandb.name task=ls_cond=${cond}_seqop=attn_nlayer=12
done

# 12-layer Transformer on in-context least squares
WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/least_squares_icl/least_squares_icl.yaml --autoname False --model.n_layer 12 --model.seq_op attn --training.train_steps ${default_train_steps} --training.scheduler_steprate ${default_scheduler_steprate} --training.learning_rate ${default_learning_rate} --wandb.name task=lsicl_seqop=attn_nlayer=12
