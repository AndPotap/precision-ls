from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2"])),
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "seq_op": merge(tstring, allowed(["attn", "base_conv"])),
    "use_mlps": merge(tboolean, required),
    "mlp_upfactor": merge(tinteger, required),
    "mlp_activ": merge(tstring, allowed(["gelu", "relu", "glu"])),
    "use_bias": merge(
        tboolean, required
    ),  # bias in linear layers of attention/MLPs and bias in LayerNorm
    "use_resid": merge(tboolean, required, default(True)),  # residual connections
    "use_seqop_ln": merge(
        tboolean, required, default(False)
    ),  # LN after sequence mixer
    "use_final_ln": merge(tboolean, required, default(False)),  # final LN
    "use_mlp_ln": merge(tboolean, required, default(False)),  # LN after MLP
    "train_proj": merge(tboolean, required, default(True)),  # train initial projection
    "train_pos": merge(tboolean, required, default(True)),  # train positional encodings
    "causal": merge(
        tboolean, required, default(True)
    ),  # causal vs. non-causal sequence mixer
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "in_dims": merge(tinteger, required),  # input dimension
    "out_dims": merge(tinteger, required),  # output dimension
    "conv_type": merge(
        tstring,
        allowed(["implicit", "short", "long"]),
    ),
    "use_softmax_for_attn": merge(
        tboolean, default(False)
    ),  # softmax vs. linear attention
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
}

SAMPLER_LIST = [
    # Data
    "gaussian",
    "least_squares",
    "ode_operator",
]

TASK_LIST = [
    # Primitives
    "identity",
    "square",
    "elementwise_multiply",
    "read",
    "linear",
    # Least squares
    "least_squares",
    "least_squares_icl",
    # Explicit gradient
    "explicit_gradient",
    # Multistep gradient descent
    "multistep_gd",
    # ODE operator
    "ode_operator_icl",
    "ode_operator_icl_final",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "data": merge(tstring, allowed(SAMPLER_LIST)),
    "data_kwargs": merge(tdict, required),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "weight_decay": merge(tfloat, default(0)),
    "scheduler_steprate": merge(tinteger, default(100000)),
    "scheduler_gamma": merge(tfloat, default(0.3)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
    "seed": merge(tinteger, default(0)),
    "optimizer": merge(
        tstring,
        allowed(["adam", "sgd", "adamw"]),
        default("adam"),
    ),
    "scheduler": merge(
        tstring,
        allowed(
            [
                "step",
                "step_threshold",
                "adaptive_threshold",
            ]
        ),
        default("step"),
    ),
    "scheduler_ema_lambda": merge(tfloat, default(0.9)),
    "scheduler_metric_threshold": merge(tfloat, default(0.9)),
    "dtype": merge(tstring, allowed(["float32", "float64"]), default("float32")),
    "l1_reg": merge(tfloat, default(0)),
    # ema
    "ema_decay": merge(tfloat, default(0)),
    "ema_lambda": merge(tfloat, default(0)),
    # track stats
    "track_grad": merge(tboolean, default(True)),
}

wandb_schema = {
    "project": merge(tstring, default("precision-ls")),
    "entity": merge(tstring, default("")),  # Add your wandb username
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
    "mode": merge(tstring, default("online")),
}

schema = {
    "autoname": merge(tboolean, default(True)),
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
