#!/bin/bash

########################################################
# Replication script for experiments with ODEs.
########################################################

# Default training parameters
default_train_steps=1000000

# # In-context ODEs with Transformer
n_layers=(4 8 12 16 24)
for n_layer in "${n_layers[@]}"; do
    WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/odes/odes_icl_final.yaml --model.n_layer ${n_layer} --autoname False --wandb.name task=odesiclfinal_nlayer=${n_layer}
done

# Explicit gradient descent with BaseConv
WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/gradient_descent/explicit_gradient.yaml --autoname False --model.use_mlps False --training.learning_rate 1e-3 --training.curriculum.dims.start 22 --training.curriculum.dims.end 22 --training.curriculum.points.start 25 --training.curriculum.points.end 25 --model.n_positions 26 --model.n_dims 22 --model.in_dims 23 --model.out_dims 22 --model.n_embd 256 --model.n_layer 4 --wandb.name task=explicitgrad_seqop=bc_setting=odes

# Least squares with Transformer
WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/least_squares/least_squares.yaml --model.n_layer 12 --model.n_dims 13 --model.in_dims 14 --model.out_dims 13 --model.n_positions 20 --training.curriculum.points.start 20 --training.curriculum.points.end 20 --training.curriculum.dims.start 13 --training.curriculum.dims.end 13 --autoname False --wandb.name task=ls_setting=odes

# In-context least squares with Transformer
WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/least_squares/least_squares_icl.yaml --model.n_layer 12 --model.n_dims 13 --model.in_dims 14 --model.out_dims 1 --model.n_positions 40 --training.curriculum.points.start 6 --training.curriculum.points.end 20 --training.curriculum.dims.start 5 --training.curriculum.dims.end 13 --autoname False --wandb.name task=lsicl_setting=odes