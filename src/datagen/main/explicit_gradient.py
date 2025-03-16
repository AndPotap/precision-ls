import torch

# Metrics
from .metrics import squared_error, mean_squared_error


# Takes data from LeastSquaresSampler
class ExplicitGradient:
    def __init__(self, setting="last", **kwargs):
        self.setting = setting

    def evaluate(self, sample_data):
        a, x, x_init, b = sample_data  # (B, L, D), (B, D, 1), (B, D), (B, L)
        B, L, D = a.shape

        if self.setting == "last":
            # Calculate gradient
            # A^T (Ax - b)
            grad = torch.einsum(
                "bld,bl->bd",
                a,  # (B, L, D)
                torch.einsum("bld,bd->bl", a, x_init) - b,  # (B, L)
            ).to(
                device=a.device
            )  # (B, D)
            grad = grad / L  # Note: this rescaling factor improves training in practice

            # Inputs: x_init, then [a_i, b_i]
            # Outputs: [grad]
            in_data = torch.zeros((B, L + 1, D + 1)).to(
                device=a.device
            )  # (B, L+1, D+1)
            in_data[:, 0, :-1] = x_init
            in_data[:, 1:, :-1] = a
            in_data[:, 1:, -1] = b

            out_data = grad  # (B, D)

        else:
            # Calculate all gradients
            # (x_init^T a_j - b_j) a_j
            residual = torch.einsum("bd,bld->bl", x_init, a) - b  # (B, L)
            grad = torch.einsum("bld,bl->bld", a, residual)  # (B, L, D)
            # \sum_{i=1}^j grad_i / j
            grad = torch.cumsum(grad, dim=1)  # (B, L, D)
            num_examples = torch.cumsum(torch.ones(L), dim=0).to(
                dtype=a.dtype, device=a.device
            )  # (L)
            grad = torch.einsum("bld,l->bld", grad, 1 / num_examples)  # (B, L, D)

            # Inputs: x_init, then [a_i, b_i]
            # Outputs: [grad[:, -1:]]
            in_data = torch.zeros((B, L + 1, D + 1))
            in_data[:, 0, :-1] = x_init
            in_data[:, 1:, :-1] = a
            in_data[:, 1:, -1] = b

            out_data = grad  # (B, L, D)

        return {
            "in": in_data,  # (B, L+1, D+1)
            "out": out_data,  # (B, D) if setting="last", (B, L, D) if setting="all"
        }

    def get_metric(self):
        if self.setting == "last":
            # Inputs (B, L+1, D+1): [a_i, b_i, x_init]
            # Outputs (B, D): [grad]
            def metric(out_pred, out):
                B, D = out.shape
                assert out_pred[:, -1, :D].shape == out.shape
                return squared_error(out_pred[:, -1:, :D], out.unsqueeze(1))

        else:
            # Inputs (B, L+1, D+1): [a_i, b_i]
            # Outputs (B, L, D): [grad]
            def metric(out_pred, out):
                B, L, D = out.shape
                assert out_pred[:, :L, :D].shape == out.shape
                return squared_error(out_pred[:, :L, :D], out)

            return metric
        return metric

    def get_training_metric(self):
        # Inputs (B, L+1, D+1): [a_i, b_i, x_init]
        # Outputs (B, D): [grad]
        if self.setting == "last":

            def training_metric(out_pred, out):
                B, D = out.shape
                assert out_pred[:, -1, :D].shape == out.shape
                return mean_squared_error(out_pred[:, -1:, :D], out.unsqueeze(1))

            return training_metric
        else:
            # Inputs (B, L+1, D+1): [a_i, b_i, x_init]
            # Outputs (B, L, D): [grad]
            def metric(out_pred, out):
                B, L, D = out.shape
                assert out_pred[:, :L, :D].shape == out.shape
                return mean_squared_error(out_pred[:, :L, :D], out)

            return metric

    def get_eval_metrics(self):
        def eval_metric(y_pred, y, x):
            return {}

        return eval_metric


# Test
if __name__ == "__main__":
    from .samplers import LeastSquaresSampler

    sampler = LeastSquaresSampler(
        n_dims=5,
        seed=0,
        device="cuda",
    )
    (a, x, x_init, b) = sampler.sample(
        n_points=20,
        batch_size=16,
        n_dims_truncated=5,
    )
    assert a.shape == (16, 20, 5)
    assert x.shape == (16, 5, 1)
    assert x_init.shape == (16, 5)
    assert b.shape == (16, 20)

    # Test ExplicitGradient
    task = ExplicitGradient(setting="last")
    task_data = task.evaluate((a, x, x_init, b))
    task_in_data = task_data["in"]
    task_out_data = task_data["out"]
    assert task_in_data.shape == (16, 21, 6)
    assert task_out_data.shape == (16, 5)

    # Check entries of task_in_data
    assert torch.allclose(task_in_data[:, 0, :-1], x_init)
    assert torch.allclose(task_in_data[:, 1:, :-1], a)
    assert torch.allclose(task_in_data[:, 1:, -1], b)

    # Check entries of task_out_data
    resid = torch.einsum("bld,bd->bl", a, x_init) - b
    grad = torch.einsum("bld,bl->bd", a, resid) / 20
    assert torch.allclose(task_out_data, grad)

    # Check metric
    perfect_pred = torch.zeros_like(task_in_data)
    perfect_pred[:, -1, :-1] = grad
    assert torch.allclose(
        task.get_metric()(perfect_pred, task_out_data),
        torch.tensor(0).to(dtype=perfect_pred.dtype, device=perfect_pred.device),
    )

    print("ExplicitGradient test passed")
