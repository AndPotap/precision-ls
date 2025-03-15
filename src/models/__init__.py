import os
from munch import Munch
import yaml

import torch

from .gpt2 import TransformerModel


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            use_mlps=conf.use_mlps,
            seq_op=conf.seq_op,
            mlp_activ=conf.mlp_activ,
            use_resid=conf.use_resid,
            use_seqop_ln=conf.use_seqop_ln,
            use_final_ln=conf.use_final_ln,
            use_mlp_ln=conf.use_mlp_ln,
            conv_type=conf.conv_type,
            causal=conf.causal,
            train_proj=conf.train_proj,
            train_pos=conf.train_pos,
            mlp_upfactor=conf.mlp_upfactor,
            in_dims=conf.in_dims,
            out_dims=conf.out_dims,
            use_softmax_for_attn=conf.use_softmax_for_attn,
        )
    else:
        raise NotImplementedError

    return model


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        filtered_state = {
            model_key: state["model_state_dict"][model_key]
            for model_key in model.state_dict().keys()
        }
        try:
            model.load_state_dict(filtered_state)
        except:
            print("Not all keys are present in state_dict, using strict=False")
            model.load_state_dict(filtered_state, strict=False)
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        try:
            model.load_state_dict(state_dict)
        except:
            print("Not all keys are present in state_dict, using strict=False")
            model.load_state_dict(state_dict, strict=False)

    return model, conf
