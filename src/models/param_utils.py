import torch

def set_param(param, setting=None, bias=True, requires_grad=False):
    if setting == "identity":
        param.weight = torch.nn.Parameter(torch.eye(*param.weight.shape))
        if bias:
            param.bias = torch.nn.Parameter(torch.zeros(*param.bias.shape))
    elif setting == "zeros":
        param.weight = torch.nn.Parameter(torch.zeros(*param.weight.shape))
        if bias:
            param.bias = torch.nn.Parameter(torch.zeros(*param.bias.shape))
    elif setting == "ones":
        param.weight = torch.nn.Parameter(torch.zeros(*param.weight.shape))
        if bias:
            param.bias = torch.nn.Parameter(torch.ones(*param.bias.shape))
    param.weight.requires_grad = requires_grad
    if bias:
        param.bias.requires_grad = requires_grad
        