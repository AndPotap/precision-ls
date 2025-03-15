import torch


# Helper function to scale the singular values of a matrix
# A: (B, L, D)
def scale_singular_values(A, l_max=2, l_min=1):
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    S *= (l_max - l_min) / (S[:, 0][:, None] - S[:, -1][:, None])
    S += l_min - S[:, -1][:, None]

    return U @ torch.diag_embed(S) @ Vh


class DataSampler:
    def __init__(self, n_dims, name=None, **kwargs):
        self.n_dims = n_dims
        self.name = name

    # Sample = (b l d)
    def sample(self, n_points, batch_size):
        raise NotImplementedError


# x: (B, L, D), samples from N(0, I)
class GaussianSampler(DataSampler):
    def __init__(
        self, n_dims, scale=1, normalize=False, seed=0, device="cuda", **kwargs
    ):
        super().__init__(n_dims, name="gaussian", **kwargs)
        self.scale = scale
        self.normalize = normalize
        self.seed = seed
        self.device = device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def sample(self, n_points, batch_size, n_dims_truncated=None):
        x = torch.randn(
            batch_size,
            n_points,
            self.n_dims,
            device=self.device,
            generator=self.generator,
        )

        # Normalize if needed
        if self.normalize:
            x = x / torch.norm(x, dim=-1, keepdim=True)

        # Scale if needed
        if self.scale != 1:
            x *= self.scale

        # Apply truncation if n_dims_truncated is set
        if n_dims_truncated is not None:
            x[:, :, n_dims_truncated:] = 0

        return x


# a: (B, L, D), samples from N(0, I)
# x: (B, D, 1), samples from N(0, I)
# x_init: (B, D), samples from N(0, I)
# b: (B, L), a @ x
class LeastSquaresSampler:
    def __init__(
        self,
        n_dims,
        seed=0,
        device="cuda",
        scale_cond=1,
        scale_target=1,
        normalize_a=False,
        normalize_x=False,
        **kwargs
    ):
        self.n_dims = n_dims
        self.seed = seed
        self.device = device
        self.scale_cond = scale_cond
        self.scale_target = scale_target
        self.normalize_a = normalize_a
        self.normalize_x = normalize_x
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def sample(self, n_points, batch_size, n_dims_truncated=None):
        # a = (B, L, D)
        a = torch.randn(
            batch_size,
            n_points,
            self.n_dims,
            device=self.device,
            generator=self.generator,
        )

        if self.normalize_a:
            a = a / torch.norm(a, dim=-1, keepdim=True)

        if self.scale_cond != 1:
            a = scale_singular_values(a, l_max=self.scale_cond)

        # x = (B, D, 1)
        x = torch.randn(
            batch_size, self.n_dims, 1, device=self.device, generator=self.generator
        )

        if self.normalize_x:
            x = x / torch.norm(x, dim=-2, keepdim=True)

        if self.scale_target != 1:
            x *= self.scale_target

        # x_init = (B, D)
        x_init = torch.randn(
            batch_size, self.n_dims, device=self.device, generator=self.generator
        )

        if self.normalize_x:
            x_init = x_init / torch.norm(x_init, dim=-1, keepdim=True)

        if self.scale_target != 1:
            x_init *= self.scale_target

        # Apply truncation if n_dims_truncated is set
        if n_dims_truncated is not None:
            a[:, :, n_dims_truncated:] = 0
            x[:, n_dims_truncated:] = 0
            x_init[:, n_dims_truncated:] = 0

        # Compute b = a @ x, (B, L)
        b = torch.einsum("bld,bd->bl", a, x.squeeze(-1)).to(device=a.device)

        return (a, x, x_init, b)
