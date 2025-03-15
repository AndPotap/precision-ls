"""
Implementation of a GPT Language Model. Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from .mixers import (
    BaseConvLayer,
    CausalSelfAttention,
)
from .param_utils import set_param


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class GLU(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj1 = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.proj2 = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.proj1(x)) * self.proj2(x)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.nlayer = 2
        self.c_fc = nn.Linear(
            config.n_embd, config.mlp_upfactor * config.n_embd, bias=config.bias
        )
        self.c_intermediates = nn.ModuleList(
            [
                nn.Linear(
                    config.mlp_upfactor * config.n_embd,
                    config.mlp_upfactor * config.n_embd,
                    bias=config.bias,
                )
                for _ in range(self.nlayer - 2)
            ]
        )
        if config.mlp_activ == "gelu":
            self.activ = nn.GELU()
        elif config.mlp_activ == "relu":
            self.activ = nn.ReLU()
        elif config.mlp_activ == "glu":
            self.activ = GLU(config.mlp_upfactor * config.n_embd)
        else:
            print(f"MLP activation {config.mlp_activ} not implemented")
        self.c_proj = nn.Linear(
            config.mlp_upfactor * config.n_embd, config.n_embd, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activ(x)
        if self.nlayer > 2:
            for _, layer in enumerate(self.c_intermediates):
                x = layer(x)
                x = self.activ(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_op = config.seq_op
        self.use_mlps = config.use_mlps
        self.use_resid = config.use_resid
        self.use_seqop_ln = config.use_seqop_ln
        self.use_mlp_ln = config.use_mlp_ln

        if self.use_seqop_ln:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        else:
            self.ln_1 = nn.Identity()

        # Sequence op: attention replacements
        if self.seq_op == "attn":
            self.seq_mixer = CausalSelfAttention(config)
        elif self.seq_op == "base_conv":
            self.seq_mixer = BaseConvLayer(config)
        elif self.seq_op == "identity":
            self.seq_mixer = nn.Identity()
        else:
            print(f"seq_op {self.seq_op} not implemented")

        if self.use_mlps:
            if self.use_mlp_ln:
                self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            else:
                self.ln_2 = nn.Identity()
            self.mlp = MLP(config)

    def _set_weights_synthetic(self, setting=None):
        if self.seq_op == "base_conv" and self.seq_mixer.base_conv.conv_type == "long":
            layer_name = self.seq_mixer.base_conv
            if setting == "square":
                set_param(layer_name.projection, setting="identity")
                layer_name.conv.filter = torch.nn.Parameter(
                    torch.zeros(*layer_name.conv.filter.shape)
                )
                layer_name.conv.filter.requires_grad = False
                layer_name.conv.filter[:, 0] = 1
            elif setting == "cumsum":
                set_param(layer_name.projection, setting="ones")
                layer_name.conv.filter = torch.nn.Parameter(
                    torch.zeros(*layer_name.conv.filter.shape)
                )
                layer_name.conv.filter.requires_grad = False
                # layer_name.conv.filter[:, :n_points] = 1
                layer_name.conv.filter[:, :] = 1
        else:
            print("Failed to set weights!")

    def forward(self, x, block_lr=1):
        if self.use_resid:
            x = x + block_lr * self.seq_mixer(self.ln_1(x))
        else:
            x = self.seq_mixer(self.ln_1(x))

        if self.use_mlps:
            if self.use_mlp_ln:
                if self.use_resid:
                    x = x + block_lr * self.mlp(self.ln_2(x))
                else:
                    x = self.mlp(self.ln_2(x))
            else:
                if self.use_resid:
                    x = x + block_lr * self.mlp(x)
                else:
                    x = self.mlp(x)

        return x


@dataclass
class GPT2Config:
    block_size: int = 101
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True  # bias in Linears and LayerNorms
    use_mlps: bool = True  # within transformer
    seq_op: str = "attn"  # as a replacement for attention. attn, hyena, autocorr
    mlp_activ: str = "gelu"  # glu
    use_resid: bool = (
        True  # whether or not to use residual connection in sequence mixer
    )
    use_seqop_ln: bool = (
        True  # whether or not to use LayerNorm within blocks, before sequence mixer
    )
    use_final_ln: bool = True  # whether or not to use LayerNorm after all blocks
    use_mlp_ln: bool = True  # whether or not to use LayerNorm in MLPs in blocks
    conv_type: str = (
        "short"  # in BaseConv, which type of conv to use. implicit (mlp), long (parameterized in real), fourier (parameterized in fourier) convs.
    )
    causal: bool = True  # whether or not to use causal sequence mixer
    mlp_upfactor: int = 4  # how much to upfactor hidden dim in MLPs

    use_softmax_for_attn: bool = (
        True  # Whether to use softmax in attn score computation.
    )

    def copy(self):
        return GPT2Config(
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias,
            use_mlps=self.use_mlps,
            seq_op=self.seq_op,
            mlp_activ=self.mlp_activ,
            use_resid=self.use_resid,
            use_seqop_ln=self.use_seqop_ln,
            use_final_ln=self.use_final_ln,
            use_mlp_ln=self.use_mlp_ln,
            conv_type=self.conv_type,
            causal=self.causal,
            mlp_upfactor=self.mlp_upfactor,
            use_softmax_for_attn=self.use_softmax_for_attn,
        )


class GPT2Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config
        self.n_layer = config.n_layer
        self.use_final_ln = config.use_final_ln

        module_dict = dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
        )

        module_dict["h"] = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        if self.use_final_ln:
            module_dict["ln_f"] = LayerNorm(config.n_embd, bias=config.bias)

        self.transformer = nn.ModuleDict(module_dict)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and module.weight.requires_grad:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _set_weights_synthetic(self, layer=None, setting=None):
        if setting == "pos":
            # Set positional embedding to zero and requires_grad=False
            set_param(self.transformer.wpe, setting="zeros", bias=False)
        else:
            assert layer is not None
            self.transformer.h[layer]._set_weights_synthetic(setting=setting)

    def forward(self, tok_emb, targets=None, temperature=1):
        device = tok_emb.device
        b = tok_emb.shape[0]
        t = tok_emb.shape[1]
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block_i, block in enumerate(self.transformer.h):
            x = block(x)

        if self.use_final_ln:
            x = self.transformer.ln_f(x)

        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=128,
        n_layer=12,
        n_head=8,
        use_mlps=True,
        seq_op="attn",
        mlp_activ="gelu",
        use_resid=True,
        use_bias=True,
        use_seqop_ln=True,
        use_final_ln=True,
        use_mlp_ln=True,
        conv_type="short",
        causal=True,
        train_proj=True,
        train_pos=True,
        mlp_upfactor=4,
        in_dims=0,
        out_dims=0,
        use_softmax_for_attn=True,
    ):
        super(TransformerModel, self).__init__()
        # Config from nanoGPT
        configuration = GPT2Config(
            block_size=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            use_mlps=use_mlps,
            seq_op=seq_op,
            mlp_activ=mlp_activ,
            use_resid=use_resid,
            bias=use_bias,
            use_seqop_ln=use_seqop_ln,
            use_final_ln=use_final_ln,
            use_mlp_ln=use_mlp_ln,
            conv_type=conv_type,
            causal=causal,
            mlp_upfactor=mlp_upfactor,
            use_softmax_for_attn=use_softmax_for_attn,
        )
        if in_dims == 0:
            in_dims = n_dims
        if out_dims == 0:
            out_dims = n_dims

        self.n_positions = n_positions
        self.n_dims = n_dims

        self._read_in = nn.Linear(in_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, out_dims)
        if not (train_proj):
            self._set_weights_synthetic(setting="proj")
        if not (train_pos):
            self._set_weights_synthetic(setting="pos")

    def _set_weights_synthetic(self, layer=None, setting=None):
        if setting == "proj":
            # Set in-projection to identity and requires_grad=False
            set_param(self._read_in, setting="identity")
            # Set out-projection to identity and requires_grad=False
            set_param(self._read_out, setting="identity")
        else:
            self._backbone._set_weights_synthetic(layer=layer, setting=setting)

    # ys = targets
    def forward(self, xs):
        embeds = self._read_in(xs)
        output = self._backbone(embeds)
        prediction = self._read_out(output)
        return prediction
