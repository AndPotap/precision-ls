#!/bin/bash

########################################################
# Replication script for experiments with the explicit
# gradient and k-th gradient descent iterate tasks.
########################################################

# Default training parameters
default_train_steps=1000000
default_scheduler_steprate=3000
default_learning_rate=1e-2

# Step scheduler vs. adaptive scheduler for explicit gradient task.
initial_lrs=(1e-2 3e-3 1e-3 3e-4)
for initial_lr in "${initial_lrs[@]}"; do
    # Step scheduler
    WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/gradient_descent/explicit_gradient.yaml --autoname False --training.learning_rate ${initial_lr} --training.scheduler step_threshold --training.scheduler_steprate ${default_scheduler_steprate} --training.scheduler_gamma 0.9 --wandb.name task=explicitgradient_lrscheduler=step_lr=${initial_lr}_steprate=${default_scheduler_steprate}_emagrad=True
    # Adaptive threshold scheduler
    WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/gradient_descent/explicit_gradient.yaml --autoname False --training.learning_rate ${initial_lr} --training.scheduler adaptive_threshold --training.scheduler_steprate ${default_scheduler_steprate} --training.scheduler_gamma 0.9 --wandb.name task=explicitgradient_lrscheduler=adaptive_lr=${initial_lr}_steprate=${default_scheduler_steprate}_emagrad=True
done

# Constant LR for explicit gradient task. No EMA over gradients.
initial_lrs=(1e-2 1e-3 1e-4 1e-5)
for initial_lr in "${initial_lrs[@]}"; do
    WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/gradient_descent/explicit_gradient.yaml --autoname False --training.learning_rate ${initial_lr} --training.scheduler step_threshold --training.scheduler_steprate ${default_scheduler_steprate} --training.train_steps ${default_train_steps} --training.ema_decay 0 --training.ema_lambda 0 --wandb.name task=explicitgradient_lrscheduler=step_lr=${initial_lr}_steprate=${default_scheduler_steprate}_emagrad=False
done

# Ablation study: effect of initial LR for explicit gradient task + step scheduler. No EMA over gradients.
initial_lrs=(1e-2 1e-3 1e-4 1e-5)
for initial_lr in "${initial_lrs[@]}"; do
    WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/gradient_descent/explicit_gradient.yaml --autoname False --training.learning_rate ${initial_lr} --training.scheduler step_threshold --training.scheduler_steprate ${default_scheduler_steprate} --training.scheduler_gamma 0.9 --training.train_steps ${default_train_steps} --training.ema_decay 0 --training.ema_lambda 0 --wandb.name task=explicitgradient_lrscheduler=step_lr=${initial_lr}_steprate=${default_scheduler_steprate}_emagrad=False
done

# Ablation study: effect of step rate for explicit gradient task + step scheduler. No EMA over gradients.
steprates=(1000 3000 10000 30000)
for steprate in "${steprates[@]}"; do
    WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/gradient_descent/explicit_gradient.yaml --autoname False --training.learning_rate ${initial_lr} --training.scheduler step_threshold --training.scheduler_steprate ${steprate} --training.scheduler_gamma 0.9 --training.train_steps ${default_train_steps} --training.ema_decay 0 --training.ema_lambda 0 --wandb.name task=explicitgradient_lrscheduler=step_lr=${initial_lr}_steprate=${steprate}_emagrad=False
done

# Ablation study: effect of MLPs and LNs on explicit gradient task.
mlps=(True False)
lns=(True False)
for mlp in "${mlps[@]}"; do
    for ln in "${lns[@]}"; do
        WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/gradient_descent/explicit_gradient.yaml --autoname False --model.use_mlps ${mlp} --model.use_seqop_ln ${ln} --training.learning_rate 1e-3 --training.scheduler step_threshold --training.scheduler_steprate 10000 --training.scheduler_gamma 0.9 --training.train_steps ${default_train_steps} --training.ema_decay 0.98 --training.ema_lambda 2 --wandb.name task=explicitgradient_lrscheduler=step_lr=1e-3_steprate=10000_emagrad=True_mlp=${mlp}_ln=${ln}
    done
done

# Explicit gradient task with linear attention.
# Warning: beginning of training can be unstable -- try decreasing the initial learning rate if this happens.
WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/gradient_descent/explicit_gradient.yaml --autoname False --model.causal False --model.use_softmax_for_attn False --model.seq_op attn --model.n_embd 256 --model.n_head 16 --model.n_layer 3 --training.learning_rate 5e-4 --training.scheduler step_threshold --training.scheduler_steprate 12500 --training.train_steps 2500001 --training.ema_decay 0.98 --training.ema_lambda 2 --wandb.name task=explicitgradient_seqop=linattn_head=16_nembd=256

# Sanity check: explicit gradient task with softmax attention.
WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/gradient_descent/explicit_gradient.yaml --autoname False --model.causal False --model.use_softmax_for_attn True --model.seq_op attn --model.n_embd 256 --model.n_head 16 --model.n_layer 3 --training.learning_rate 5e-4 --training.scheduler step_threshold --training.scheduler_steprate 12500 --training.train_steps 2500001 --training.ema_decay 0.98 --training.ema_lambda 2 --wandb.name task=explicitgradient_seqop=attn_head=16_nembd=256

# K-th gradient descent iterate task. Here, k=4 with the adaptive threshold scheduler.
# Warning: beginning of training can be unstable -- try decreasing the initial learning rate if this happens.
WANDB__SERVICE_WAIT=300 python src/train.py --config src/conf/gradient_descent/multistep_step=4.yaml --autoname False --training.learning_rate 1e-2 --training.scheduler step_threshold --training.scheduler_steprate 3000 --training.scheduler_gamma 0.9 --training.train_steps 1000000 --training.ema_decay 0 --training.ema_lambda 0 --wandb.name task=multistep_step=4_lrscheduler=step_lr=1e-2_steprate=3000
