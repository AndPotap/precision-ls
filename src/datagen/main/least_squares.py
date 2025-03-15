import torch

# Metrics
from .metrics import squared_error, mean_squared_error


# Takes data from LeastSquaresSampler
class LeastSquares:
    def __init__(self, **kwargs):
        pass

    def evaluate(self, sample_data):
        a, x, _, b = sample_data  # (B, L, D), (B, D, 1), (B, D), (B, L)
        B, L, D = x.shape

        # Input to model: [a_i, b_i], (B, L, D+1)
        # Output: x, (B, 1, D)
        in_data = torch.cat([a, b.unsqueeze(-1)], dim=-1)  # (B, L, D+1)
        out_data = x.transpose(-1, -2)  # (B, 1, D)

        return {
            "in": in_data,
            "out": out_data,
        }

    def get_metric(self):
        def metric(out_pred, out):
            B, _, D = out.shape
            assert out_pred[:, -1:, :D].shape == out.shape
            return squared_error(out_pred[:, -1:, :D], out)

        return metric

    def get_training_metric(self):
        def training_metric(out_pred, out):
            B, _, D = out.shape
            assert out_pred[:, -1:, :D].shape == out.shape
            return mean_squared_error(out_pred[:, -1:, :D], out)

        return training_metric

    def get_eval_metrics(self):
        def eval_metric(out_pred, out, in_data):
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

    # Test LeastSquares
    task = LeastSquares()
    task_data = task.evaluate((a, x, x_init, b))
    task_in_data = task_data["in"]
    task_out_data = task_data["out"]
    assert task_in_data.shape == (16, 20, 6)
    assert task_out_data.shape == (16, 1, 5)

    # Check entries of task_in_data
    assert torch.allclose(task_in_data[:, :, :-1], a)
    assert torch.allclose(task_in_data[:, :, -1], b)

    # Check entries of task_out_data
    assert torch.allclose(task_out_data.squeeze(1), x.squeeze(-1))

    # Check metric
    perfect_pred = torch.zeros_like(task_in_data)
    perfect_pred[:, -1, :-1] = x.squeeze(-1)
    assert torch.allclose(
        task.get_metric()(perfect_pred, task_out_data),
        torch.tensor(0).to(dtype=perfect_pred.dtype, device=perfect_pred.device),
    )

    print("LeastSquares test passed")
