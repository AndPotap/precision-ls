"""
Implementation of EMA-based gradient filtering.
Source: https://github.com/ironjr/grokfast/blob/main/grokfast.py
"""

from collections import deque
from typing import Dict, Optional, Literal, Tuple
import torch
import torch.nn as nn


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal["mean", "sum"] = "mean",
    warmup: bool = True,
    trigger: bool = False,
) -> Tuple[Dict[str, deque], Dict[str, torch.Tensor]]:
    if grads is None:
        grads = {
            n: deque(maxlen=window_size)
            for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach())

            # Modify the gradients.
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, deque]]:
    if grads is None:
        grads = {
            n: p.grad.data.detach()
            for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data * (1 / (1 + lamb)) + grads[n] * (
                lamb / (1 + lamb)
            )

    return grads


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    grad_history: Optional[Dict[str, deque]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
    history_size: int = 64,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, deque]]:
    if grads is None:
        grads = {
            n: p.grad.data.detach()
            for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    if grad_history is None:
        # Collect gradients
        batch_gradients = []
        for n, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                batch_gradients.append(p.grad.view(-1).detach().clone())
        batch_gradients = torch.cat(batch_gradients)

        grad_history = deque([batch_gradients], maxlen=history_size)

    update_gradients = []
    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            # Update EMA of gradients
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            # Compute the filtered gradient (the actual update vector)
            update_vector = p.grad.data + grads[n] * lamb
            # Update the gradient history with the update vector
            update_gradients.append(update_vector.view(-1).detach().clone())
            # Apply the filtered gradient update to the model parameters
            p.grad.data = update_vector
    update_gradients = torch.cat(update_gradients)
    grad_history.append(update_gradients)

    return grads, grad_history


def grad_history_to_tensor(grad_history: deque) -> torch.Tensor:
    # Convert the deque to a list of tensors
    grad_list = list(grad_history)
    # Stack the tensors into a single [B, num_params] tensor
    grad_tensor = torch.stack(grad_list)
    return grad_tensor
