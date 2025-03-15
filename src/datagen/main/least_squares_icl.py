import torch

# Metrics
from .metrics import squared_error, mean_squared_error


# Takes data from GradientDescentSampler
# Settings: "concat" or "neighboring"
class LeastSquaresICL:
    def __init__(self, setting="neighboring", **kwargs):
        self.setting = setting

    def evaluate(self, sample_data):
        a, x, _, b = sample_data  # (B, L, D), (B, D, 1), (B, D), (B, L)
        B, L, D = a.shape

        if self.setting == "neighboring":
            # Inputs: [a_i, b_i]
            # Outputs: [b_i]
            in_data = torch.zeros((B, 2 * L - 1, D + 1)).to(
                device=a.device
            )  # (B, 2L-1, D+1)
            in_data[:, ::2, :-1] = a
            in_data[:, 1::2, -1] = b[:, :-1]

            out_data = b.unsqueeze(-1)  # (B, L, 1)

        elif self.setting == "concat":
            # Inputs: [a_i, b_i]
            # Outputs: [b_i]
            in_data = torch.zeros((B, L, D + 1)).to(device=a.device)
            in_data[:, :, :-1] = a
            in_data[:, :-1, -1] = b[:, :-1]

            out_data = b[:, -1:].unsqueeze(-1)  # (B, 1, 1)

        return {
            "in": in_data,
            "out": out_data,
        }

    def get_metric(self):
        if self.setting == "neighboring":

            def metric(out_pred, out):
                assert out_pred[:, ::2, -1:].shape == out.shape
                return squared_error(out_pred[:, ::2, -1:], out)

        elif self.setting == "concat":

            def metric(out_pred, out):
                assert out_pred[:, -1:, -1:].shape == out.shape
                return squared_error(out_pred[:, -1:, -1:], out)

        return metric

    def get_training_metric(self):
        if self.setting == "neighboring":

            def training_metric(out_pred, out):
                assert out_pred[:, ::2, -1:].shape == out.shape
                return mean_squared_error(out_pred[:, ::2, -1:], out)

        elif self.setting == "concat":

            def training_metric(out_pred, out):
                assert out_pred[:, -1:, -1:].shape == out.shape
                return mean_squared_error(out_pred[:, -1:, -1:], out)

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

    # Test LeastSquaresICL: setting = "neighboring"
    task = LeastSquaresICL(setting="neighboring")
    task_data = task.evaluate((a, x, x_init, b))
    task_in_data = task_data["in"]
    task_out_data = task_data["out"]
    assert task_in_data.shape == (16, 39, 6)
    assert task_out_data.shape == (16, 20, 1)

    # Check entries of task_in_data
    assert torch.allclose(task_in_data[:, ::2, :-1], a)
    assert torch.allclose(task_in_data[:, 1::2, -1], b[:, :-1])

    # Check entries of task_out_data
    assert torch.allclose(task_out_data.squeeze(-1), b)

    # Check metric
    perfect_pred = torch.zeros_like(task_in_data)
    perfect_pred[:, ::2, -1] = b
    assert torch.allclose(
        task.get_metric()(perfect_pred, task_out_data),
        torch.tensor(0).to(dtype=perfect_pred.dtype, device=perfect_pred.device),
    )

    # Test LeastSquaresICL: setting = "concat"
    task = LeastSquaresICL(setting="concat")
    task_data = task.evaluate((a, x, x_init, b))
    task_in_data = task_data["in"]
    task_out_data = task_data["out"]
    assert task_in_data.shape == (16, 20, 6)
    assert task_out_data.shape == (16, 1, 1)

    # Check entries of task_in_data
    assert torch.allclose(task_in_data[:, :, :-1], a)
    assert torch.allclose(task_in_data[:, :-1, -1], b[:, :-1])

    # Check entries of task_out_data
    assert torch.allclose(task_out_data.squeeze(-1), b[:, -1:])

    # Check metric
    perfect_pred = torch.zeros_like(task_in_data)
    perfect_pred[:, -1:, -1] = b[:, -1:]
    assert torch.allclose(
        task.get_metric()(perfect_pred, task_out_data),
        torch.tensor(0).to(dtype=perfect_pred.dtype, device=perfect_pred.device),
    )

    print("LeastSquaresICL test passed")
