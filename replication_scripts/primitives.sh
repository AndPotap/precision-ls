#!/bin/bash

########################################################
# Replication script for experiments with numerical primitives.
########################################################

# Default training parameters
default_train_steps=1000000
default_scheduler_steprate=10000
default_learning_rate=1e-3

# Transformer and BaseConv, with and without LNs. Sweep number of layers.
# Primitives: read, linear, multiply
primitives=(read linear multiply)
seq_ops=(attn base_conv)
use_lns=(True False)
n_layers=(1 2 4 8)
for primitive in ${primitives[@]}; do
    for seq_op in ${seq_ops[@]}; do
        for use_ln in ${use_lns[@]}; do
            for n_layer in ${n_layers[@]}; do
                WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/primitives/${primitive}.yaml --autoname False --model.n_layer ${n_layer} --model.seq_op ${seq_op} --model.use_seqop_ln ${use_ln} --training.train_steps ${default_train_steps} --training.scheduler_steprate ${default_scheduler_steprate} --training.learning_rate ${default_learning_rate} --wandb.name task=${primitive}_seqop=${seq_op}_seqopln=${use_ln}_nlayer=${n_layer}
            done
        done
    done
done

# Transformer: sweep hidden dimension.
hdims=(32 64 128 256)
for hdim in ${hdims[@]}; do
    WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/primitives/multiply.yaml --autoname False --model.n_layer 2 --model.seq_op attn --model.use_seqop_ln True --model.n_embd ${hdim} --model.n_head 1 --training.train_steps ${default_train_steps} --training.scheduler_steprate ${default_scheduler_steprate} --training.learning_rate ${default_learning_rate} --wandb.name task=multiply_seqop=attn_hdim=${hdim}
done

# Transformer: sweep number of heads.
n_heads=(1 2 4 8)
for n_head in ${n_heads[@]}; do
    WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/primitives/multiply.yaml --autoname False --model.n_layer 2 --model.seq_op attn --model.use_seqop_ln True --model.n_embd 256 --model.n_head ${n_head} --training.train_steps ${default_train_steps} --training.scheduler_steprate ${default_scheduler_steprate} --training.learning_rate ${default_learning_rate} --wandb.name task=multiply_seqop=attn_nhead=${n_head}
done

# Transformer: sweep MLP width
mlp_upfactors=(1 2 4 8)
for mlp_upfactor in ${mlp_upfactors[@]}; do
    WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/primitives/multiply.yaml --autoname False --model.n_layer 2 --model.seq_op attn --model.use_seqop_ln True --model.n_embd 256 --model.n_head 1 --model.mlp_upfactor ${mlp_upfactor} --training.train_steps ${default_train_steps} --training.scheduler_steprate ${default_scheduler_steprate} --training.learning_rate ${default_learning_rate} --wandb.name task=multiply_seqop=attn_mlpupfactor=${mlp_upfactor}
done

# Transformer and BaseConv: sweep number of training iterations
seq_ops=(attn base_conv)
train_steps=(100000 1000000 10000000 100000000)
for seq_op in ${seq_ops[@]}; do
    for train_step in ${train_steps[@]}; do
        # Scale scheduler_steprate accordingly
        scheduler_steprate=$((train_step / 100))
        WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/primitives/multiply.yaml --autoname False --model.n_layer 2 --model.seq_op ${seq_op} --model.use_seqop_ln False --model.n_embd 256 --model.n_head 1 --training.train_steps ${train_step} --training.scheduler_steprate ${scheduler_steprate} --training.learning_rate ${default_learning_rate} --wandb.name task=multiply_seqop=${seq_op}_trainsteps=${train_step}
    done
done
