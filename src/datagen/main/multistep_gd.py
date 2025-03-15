import torch

# Metrics
from .metrics import squared_error, mean_squared_error


# Takes data from LeastSquaresSampler
class MultistepGradientDescent:
    def __init__(self, num_iters=1, step_size=0.5, **kwargs):
        self.num_iters = num_iters
        self.step_size = step_size

    def evaluate(self, sample_data):
        a, x, x_init, b = sample_data  # (B, L, D), (B, D, 1), (B, D), (B, L)
        B, L, D = a.shape
        x_final = x_init.clone()  # (B, D)

        for iter_i in range(self.num_iters):
            # Calculate gradient
            # A^T (Ax - b) / n
            grad = torch.einsum(
                "bld,bl->bd",
                a,  # (B, L, D)
                torch.einsum("bld,bd->bl", a, x_final) - b,  # (B, L)
            ).to(
                device=a.device
            )  # (B, D)

            # Update x_final
            x_final -= self.step_size * grad / L  # (B, D)

        # Inputs: [a_i, b_i], then x_init
        # Outputs: [x_final]
        in_data = torch.zeros((B, L + 1, D + 1)).to(device=a.device)  # (B, L+1, D+1)
        in_data[:, :-1, :-1] = a
        in_data[:, :-1, -1] = b
        in_data[:, -1, :-1] = x_init

        out_data = x_final.unsqueeze(1)  # (B, 1, D)

        return {
            "in": in_data,  # (B, L+1, D+1)
            "out": out_data,  # (B, 1, D)
        }

    def get_metric(self):
        # Inputs (B, L+1, D+1): [a_i, b_i, x_init]
        # Outputs (B, D): [x_final]
        def metric(out_pred, out):
            B, L, D = out.shape
            assert out_pred[:, -1:, :D].shape == out.shape
            return squared_error(out_pred[:, -1:, :D], out)

        return metric

    def get_training_metric(self):
        # Inputs (B, L+1, D+1): [a_i, b_i, x_init]
        # Outputs (B, D): [x_final]
        def metric(out_pred, out):
            B, L, D = out.shape
            assert out_pred[:, -1:, :D].shape == out.shape
            return mean_squared_error(out_pred[:, -1:, :D], out)

        return metric

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

    # Test MultistepGradientDescent
    task = MultistepGradientDescent(
        num_iters=1,
        step_size=0.5,
    )
    task_data = task.evaluate((a, x, x_init, b))
    task_in_data = task_data["in"]
    task_out_data = task_data["out"]
    assert task_in_data.shape == (16, 21, 6)
    assert task_out_data.shape == (16, 1, 5)

    # Check entries of task_in_data
    assert torch.allclose(task_in_data[:, -1, :-1], x_init)
    assert torch.allclose(task_in_data[:, :-1, :-1], a)
    assert torch.allclose(task_in_data[:, :-1, -1], b)

    # Check entries of task_out_data
    resid = torch.einsum("bld,bd->bl", a, x_init) - b
    grad = torch.einsum("bld,bl->bd", a, resid) / 20 * 0.5
    assert torch.allclose(task_out_data.squeeze(1), x_init - grad)

    # Check metric
    perfect_pred = torch.zeros_like(task_in_data)
    perfect_pred[:, -1, :-1] = x_init - grad
    assert torch.all(
        task.get_metric()(perfect_pred, task_out_data)
        == torch.tensor(0).to(dtype=perfect_pred.dtype, device=perfect_pred.device)
    )

    print("MultistepGradientDescent test passed")
