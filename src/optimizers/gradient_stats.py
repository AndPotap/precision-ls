import torch
from torch.nn.functional import cosine_similarity

"""
Compute gradient statistics for a given model and criterion.
Samples n_batches of batch_size samples from the data_sampler and computes the gradient statistics.
Args:
    model (nn.Module): Model to compute gradients for.
    criterion (nn.Module): Criterion to compute gradients for.
    n_batches (int): Number of batches to sample.
    batch_size (int): Batch size to use.
    data_sampler (callable): Function to sample data from.
Returns:
    torch.Tensor: List of gradients. Shape: [n_batches, n_params].
"""


def get_current_gradients(
    model, criterion, n_batches=64, batch_size=64, data_sampler=None
):
    model.train()
    gradients = []

    for _ in range(n_batches):
        # Sample
        xs, ys = data_sampler(batch_size=batch_size)
        # Forward
        model.zero_grad()
        pred = model(xs)
        loss = criterion(pred, ys)
        # Backward
        loss.backward()

        # Collect gradients
        batch_gradients = []
        for param in model.parameters():
            if param.grad is not None:
                batch_gradients.append(param.grad.view(-1))
        batch_gradients = torch.cat(batch_gradients)
        gradients.append(batch_gradients)

    gradients = torch.stack(gradients)

    return gradients


"""
Compute gradient statistics for a given model and criterion.
Samples n_batches of batch_size samples from the data_sampler and computes the gradient statistics.
Store separate gradients for each parameter.
Args:
    model (nn.Module): Model to compute gradients for.
    criterion (nn.Module): Criterion to compute gradients for.
    n_batches (int): Number of batches to sample.
    batch_size (int): Batch size to use.
    data_sampler (callable): Function to sample data from.
Returns:
    dict of torch.Tensor: {param_name : list of gradients (shape: [n_batches, n_params])}.
"""


def get_current_gradients_perparam(
    model, criterion, n_batches=64, batch_size=64, data_sampler=None
):
    model.train()

    # Init dictionary to store gradients for each parameter
    gradients = {name: [] for name, _ in model.named_parameters()}

    for _ in range(n_batches):
        # Sample
        xs, ys = data_sampler(batch_size=batch_size)
        # Forward
        model.zero_grad()
        pred = model(xs)
        loss = criterion(pred, ys)
        # Backward
        loss.backward()

        # Collect gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name].append(param.grad.view(-1))

    # Stack gradients for each parameter
    for name, _ in model.named_parameters():
        gradients[name] = torch.stack(gradients[name])

    return gradients


"""
Compute gradient statistics for a given list of gradients.
Args:
    gradients (torch.Tensor): List of gradients to compute statistics for. Shape: [n_gradients, n_params].
    prefix (str): Prefix to add to the computed statistics (key names).
Returns:
    dict: Dictionary containing computed statistics.
"""


def compute_gradient_stats(gradients, prefix=""):

    # Compute cosine similarities between consecutive gradients
    cosine_sims = []
    for i in range(1, len(gradients)):
        cosine_sim = cosine_similarity(
            gradients[i - 1].unsqueeze(0), gradients[i].unsqueeze(0)
        ).item()
        cosine_sims.append(cosine_sim)
    cosine_sims = torch.tensor(cosine_sims)

    # Compute the mean of cosine similarities (directional similarity)
    cosine_mean = torch.mean(cosine_sims).item()
    # Compute the variance of cosine similarities (directional variance)
    cosine_variance = torch.var(cosine_sims, unbiased=True).item()

    # Compute gradient L1 norm
    gradient_l1_norm = torch.norm(gradients, p=1, dim=1)
    l1_metric = gradient_l1_norm.mean().item()
    # Compute gradient L2 norm
    gradient_l2_norm = torch.norm(gradients, p=2, dim=1)
    l2_metric = gradient_l2_norm.mean().item()
    # Compute mean of gradient entries
    gradient_mean = torch.mean(torch.abs(gradients), dim=0)
    mean_metric = gradient_mean.mean().item()
    # Compute gradient variance
    gradient_variance = torch.var(gradients, dim=0, unbiased=True)
    variance_metric = gradient_variance.mean().item()

    if prefix is not None and prefix != "":
        prefix = f"{prefix}_" if prefix[-1] != "_" else prefix

    return {
        f"{prefix}gradient_l1_norm": l1_metric,
        f"{prefix}gradient_l2_norm": l2_metric,
        f"{prefix}gradient_mean": mean_metric,
        f"{prefix}gradient_variance": variance_metric,
        f"{prefix}gradient_cosine_mean": cosine_mean,
        f"{prefix}gradient_cosine_variance": cosine_variance,
    }


"""
Compute gradient statistics for a given model and criterion.
Samples n_batches of batch_size samples from the data_sampler and computes the gradient statistics.
Args:
    model (nn.Module): Model to compute gradients for.
    criterion (nn.Module): Criterion to compute gradients for.
    n_batches (int): Number of batches to sample.
    batch_size (int): Batch size to use.
    data_sampler (callable): Function to sample data from.
Returns:
    dict: Dictionary containing computed statistics.
"""


def compute_current_gradient_stats(
    model, criterion, n_batches=64, batch_size=64, data_sampler=None
):
    gradients = get_current_gradients(
        model,
        criterion,
        n_batches=n_batches,
        batch_size=batch_size,
        data_sampler=data_sampler,
    )
    stats = compute_gradient_stats(gradients)
    return stats


"""
Compute gradient statistics between a given list of gradients and a ground truth gradient.
Args:
    gradients (torch.Tensor): List of gradients to compute statistics for. Shape: [n_gradients, n_params].
    gt_gradient (torch.Tensor): Ground truth gradient to compare with. Shape: [n_params].
    prefix (str): Prefix to add to the computed statistics (key names).
Returns:
    dict: Dictionary containing computed statistics.
"""


def compute_gradient_stats_gt(gradients, gt_gradient, prefix=""):

    # Compute cosine similarities between consecutive gradients
    cosine_sims = []
    for i in range(len(gradients)):
        cosine_sim = cosine_similarity(
            gradients[i].unsqueeze(0), gt_gradient.unsqueeze(0)
        ).item()
        cosine_sims.append(cosine_sim)
    cosine_sims = torch.tensor(cosine_sims)

    # Compute the mean of cosine similarities (directional similarity)
    cosine_mean = torch.mean(cosine_sims).item()
    # Compute the variance of cosine similarities (directional variance)
    cosine_variance = torch.var(cosine_sims, unbiased=True).item()

    # Compute mean of gradient entry-wise differences
    gradient_diffs = (gradients - gt_gradient).abs()

    gradient_mean = torch.mean(gradient_diffs, dim=0)
    mean_metric = gradient_mean.mean().item()
    # Compute gradient variance
    gradient_variance = torch.var(gradient_diffs, dim=0, unbiased=True)
    variance_metric = gradient_variance.mean().item()

    if prefix is not None and prefix != "":
        prefix = f"{prefix}_" if prefix[-1] != "_" else prefix

    return {
        f"{prefix}gradient_mean": mean_metric,
        f"{prefix}gradient_variance": variance_metric,
        f"{prefix}gradient_cosine_mean": cosine_mean,
        f"{prefix}gradient_cosine_variance": cosine_variance,
    }
