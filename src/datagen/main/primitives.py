import torch

from .metrics import (
    squared_error,
    mean_squared_error,
    mean_relative_error,
    mean_relative_l2_error,
)
from .tasks import Task

"""
# Inputs: x = (b l d)
# Outputs: y = (b l d)
"""


class Primitive(Task):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def evaluate(self, x):
        return {
            "in": x,
            "out": self.out(x),
        }

    def out(self, x):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

    @staticmethod
    def get_eval_metrics():
        def metric(y_pred, y, x):
            return {
                "mse": mean_squared_error(y_pred, y),
                "rel": mean_relative_error(y_pred, y),
                "rel_l2": mean_relative_l2_error(y_pred, y),
            }

        return metric


# y = x
class Identity(Primitive):
    def __init__(self, **kwargs):
        super().__init__(name="identity", **kwargs)

    def out(self, x):
        return x


# y[:, i, :] = x[:, i, :]**2
class Square(Primitive):
    def __init__(self, **kwargs):
        super().__init__(name="square", **kwargs)

    def out(self, x):
        return x**2


# y[:, :, :D//2] = x[:, :, :D//2] * x[:, :, D//2:]
class ElementwiseMultiply(Primitive):
    def __init__(self, **kwargs):
        super().__init__(name="elementwise_multiply", **kwargs)

    def out(self, x):
        (B, L, D) = x.shape
        assert D % 2 == 0
        y = torch.zeros((B, L, D // 2)).to(device=x.device, dtype=x.dtype)
        y = x[:, :, : D // 2] * x[:, :, D // 2 :]
        return y

    def get_metric(self):
        def metric(y_pred, y):
            return squared_error(y_pred[:, :, : y.shape[-1]], y)

        return metric

    def get_training_metric(self):
        def training_metric(y_pred, y):
            return mean_squared_error(y_pred[:, :, : y.shape[-1]], y)

        return training_metric

    def get_eval_metrics(self):
        def metric(y_pred, y, x):
            y_pred_out = y_pred[:, :, : y.shape[-1]]
            return {
                "mse": mean_squared_error(y_pred_out, y),
                "rel": mean_relative_error(y_pred_out, y),
                "rel_l2": mean_relative_l2_error(y_pred_out, y),
            }

        return metric


class Read(Primitive):
    def __init__(self, n_points, seed=0, device="cuda", **kwargs):
        super().__init__(name="read", **kwargs)
        self.seed = seed
        self.device = device
        self.n_points = n_points
        self.rng = torch.Generator(device=self.device).manual_seed(self.seed)
        # Sample integers i, j < n_points
        self.ij = torch.randint(
            0, self.n_points, (2,), device=self.device, generator=self.rng
        )
        # For causal models, need i <= j
        self.i = torch.where(self.ij[0] <= self.ij[1], self.ij[0], self.ij[1])
        self.j = torch.where(self.ij[0] <= self.ij[1], self.ij[1], self.ij[0])

    def out(self, x):
        y = x.clone()
        y[:, self.j] = x[:, self.i]
        return y


class Linear(Primitive):
    def __init__(self, n_dims, seed=0, device="cuda", sparsity=None, **kwargs):
        super().__init__(name="linear", **kwargs)
        self.seed = seed
        self.device = device
        self.n_dims = n_dims
        self.sparsity = sparsity if sparsity is not None else n_dims
        self.rng = torch.Generator(device=self.device).manual_seed(self.seed)

        self.c = 3 * torch.randn(self.n_dims, device=self.device, generator=self.rng)
        if self.sparsity is not None:
            num_zero = self.n_dims - self.sparsity
            if num_zero > 0:
                zero_indices = torch.randperm(self.n_dims, generator=self.rng)[
                    :num_zero
                ]
                self.c[zero_indices] = 0

    def out(self, x):
        assert x.shape[-1] == self.n_dims
        y = torch.einsum("bld,d->bl", x, self.c)
        return y.unsqueeze(-1)

    def get_metric(self):
        def metric(y_pred, y):
            assert y_pred[:, :, :1].shape == y.shape
            return squared_error(y_pred[:, :, :1], y)

        return metric

    def get_training_metric(self):
        def training_metric(y_pred, y):
            assert y_pred[:, :, :1].shape == y.shape
            return mean_squared_error(y_pred[:, :, :1], y)

        return training_metric

    def get_eval_metrics(self):
        def metric(y_pred, y, x):
            y_pred_out = y_pred[:, :, :1]
            return {
                "mse": mean_squared_error(y_pred_out, y),
                "rel": mean_relative_error(y_pred_out, y),
                "rel_l2": mean_relative_l2_error(y_pred_out, y),
            }

        return metric


# Test
if __name__ == "__main__":
    from .samplers import GaussianSampler

    sampler = GaussianSampler(
        n_dims=20,
        seed=0,
        device="cuda",
    )
    x = sampler.sample(
        n_points=40,
        batch_size=16,
    )
    assert x.shape == (16, 40, 20)

    # Test Square
    task = Square()
    task_data = task.evaluate(x)
    task_in_data = task_data["in"]
    task_out_data = task_data["out"]
    assert task_in_data.shape == (16, 40, 20)
    assert task_out_data.shape == (16, 40, 20)
    perfect_pred_square = x**2
    assert torch.allclose(
        task.get_metric()(perfect_pred_square, task_out_data),
        torch.tensor(0).to(
            dtype=perfect_pred_square.dtype, device=perfect_pred_square.device
        ),
    )

    # Test ElementwiseMultiply
    task = ElementwiseMultiply()
    task_data = task.evaluate(x)
    task_in_data = task_data["in"]
    task_out_data = task_data["out"]
    assert task_in_data.shape == (16, 40, 20)
    assert task_out_data.shape == (16, 40, 10)
    perfect_pred_multiply = x[:, :, :10] * x[:, :, 10:]
    assert torch.allclose(
        task.get_metric()(perfect_pred_multiply, task_out_data),
        torch.tensor(0).to(
            dtype=perfect_pred_multiply.dtype, device=perfect_pred_multiply.device
        ),
    )

    # Test Read
    task = Read(n_points=40, seed=0)
    task_data = task.evaluate(x)
    task_in_data = task_data["in"]
    task_out_data = task_data["out"]
    assert task_in_data.shape == (16, 40, 20)
    assert task_out_data.shape == (16, 40, 20)
    perfect_pred_read = x.clone()
    perfect_pred_read[:, task.j] = x[:, task.i]
    assert torch.allclose(
        task.get_metric()(perfect_pred_read, task_out_data),
        torch.tensor(0).to(
            dtype=perfect_pred_read.dtype, device=perfect_pred_read.device
        ),
    )

    # Test Linear
    task = Linear(n_dims=20, seed=0, device="cuda")
    task_data = task.evaluate(x)
    task_in_data = task_data["in"]
    task_out_data = task_data["out"]
    assert task_in_data.shape == (16, 40, 20)
    assert task_out_data.shape == (16, 40, 1)
    perfect_pred_affine = torch.einsum("bld,d->bl", x, task.c).unsqueeze(-1)
    assert torch.allclose(
        task.get_metric()(perfect_pred_affine, task_out_data),
        torch.tensor(0).to(
            dtype=perfect_pred_affine.dtype, device=perfect_pred_affine.device
        ),
    )

    print("Primitives test passed")
