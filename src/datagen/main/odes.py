import numpy as np
import torch

from .samplers import DataSampler
from .tasks import Task
from .metrics import squared_error, mean_squared_error

from .util.odes import (
    sample_gp,
    sample_params,
    interpolate_cheb,
    pseudospectral_solve_batch,
)


# Sample parameters for ODE, initial condition, and forcing function
# Solve ODE on [-1, 1] with pseudospectral method
class ODEOperatorSampler(DataSampler):
    def __init__(
        self,
        n_dims,
        c_sampling="equispaced",  # equispaced, cheb, randt
        eqn_class=0,  # 0, 1, 2
        operator_scale=1,
        gp_length=1,
        u0_scale=1,
        seed=0,
        device="cuda",
        **kwargs,
    ):
        super().__init__(n_dims=n_dims, name="ode_operator", **kwargs)
        self.c_sampling = c_sampling
        self.gp_length = gp_length
        self.u0_scale = u0_scale

        # Chebyshev
        self.n_cheb = 40
        self.to_cheb = lambda x: np.cos(np.pi * x)
        self.t_cheb = self.to_cheb(np.linspace(0, 1, self.n_cheb + 1))

        # Equation
        self.eqn_class = eqn_class
        self.operator_scale = operator_scale

        self.seed = seed
        self.device = device
        self.generator = np.random.default_rng(seed=seed)

    def sample(self, n_points, batch_size, n_dims_truncated=None):
        n_samples = n_dims_truncated

        # Sample t_eval
        if self.c_sampling == "equispaced":
            t_eval = np.linspace(-1, 1, n_samples)
        elif self.c_sampling == "cheb":
            t_eval = self.to_cheb(np.linspace(0, 1, n_samples))
        elif self.c_sampling == "randt":
            t_eval = self.generator.uniform(-1, 1, n_samples)
        t = np.concatenate([self.t_cheb, t_eval])

        # Sample c
        c = sample_gp(
            t,
            shape=(batch_size, n_points),
            variance=self.gp_length,
            rng=self.generator,
        )  # (batch_size, n_points, len(t))
        c_cheb = c[:, :, : self.n_cheb + 1]  # (batch_size, n_points, n_cheb+1)
        c_eval = c[:, :, self.n_cheb + 1 :]  # (batch_size, n_points, n_samples)

        # Sample u0
        u0 = self.generator.uniform(
            -self.u0_scale, self.u0_scale, (batch_size, n_points)
        )  # (batch_size, n_points)

        # Sample equation parameters
        if self.eqn_class in [0, 1, 2]:
            a1, a2, a3, a4 = sample_params(
                eqn_class=self.eqn_class,
                size=(batch_size),
                scale=self.operator_scale,
                rng=self.generator,
            )
        else:
            print(f"eqn_class {self.eqn_class} not implemented in ODEOperator")
            raise NotImplementedError

        # Solve for u pseudospectrally on [-1, 1]
        if self.eqn_class in [0, 1, 2]:
            u_cheb = pseudospectral_solve_batch(
                np.tile(a1.reshape(-1, 1), (1, n_points)),
                np.tile(a2.reshape(-1, 1), (1, n_points)),
                np.tile(a3.reshape(-1, 1), (1, n_points)),
                np.tile(a4.reshape(-1, 1), (1, n_points)),
                u0,
                c_cheb,
                c_cheb,
            )  # (batch_size, n_points, n_cheb+1)

        # Raw task data
        task_data = {}
        task_data["c_cheb"] = c_cheb  # (batch_size, n_points, n_cheb+1)
        task_data["c_eval"] = c_eval  # (batch_size, n_points, n_samples)
        task_data["u0"] = u0  # (batch_size, n_points)
        task_data["operator_params"] = (a1, a2, a3, a4)
        task_data["u_cheb"] = u_cheb  # (batch_size, n_points, N_cheb+1)

        return task_data


class ODEOperatorICL(Task):
    def __init__(self, noise_variance=0, seed=0, device="cuda", **kwargs):
        super().__init__(name="ode_operator", **kwargs)
        self.device = device
        self.rng = np.random.default_rng(seed=seed)
        self.noise_variance = noise_variance

    def evaluate(self, task_data):
        c_eval = task_data["c_eval"]  # (batch_size, n_points, n_samples)
        B, L, D = c_eval.shape

        # Chebyshev interpolate onto sampled points
        t_eval = self.rng.uniform(-1, 1, B)
        u_cheb = task_data["u_cheb"]
        u_eval = interpolate_cheb(u_cheb, t_eval)  # (batch_size, n_points, n_samples)
        u_eval = u_eval[np.arange(B), :, np.arange(B)]  # (batch_size, n_points)

        if self.noise_variance != 0:
            u_ood = u_eval + self.noise_variance * self.rng.standard_normal(
                u_eval.shape
            )
        else:
            u_ood = u_eval

        # Inputs: [(c, u0, 1), u_ood]
        # Outputs: [u_ood]
        in_data = torch.zeros((B, 2 * L - 1, D + 3)).to(
            device=self.device
        )  # (B, 2L-1, D+3)
        in_data[:, ::2, :-3] = torch.tensor(c_eval)
        in_data[:, ::2, -3] = torch.tensor(task_data["u0"])
        in_data[:, ::2, -2] = 1
        in_data[:, 1::2, -1] = torch.tensor(u_ood)[:, :-1]

        out_data = (
            torch.tensor(u_eval).to(device=self.device).unsqueeze(-1)
        )  # (B, L, 1)

        return {
            "in": in_data,  # (B, 2L-1, D+3)
            "out": out_data,  # (B, L, 1)
        }

    def get_metric(self):
        def metric(out_pred, out):
            assert out_pred[:, ::2, -1:].shape == out.shape
            return squared_error(out_pred[:, ::2, -1:], out)

        return metric

    def get_training_metric(self):
        def training_metric(out_pred, out):
            assert out_pred[:, ::2, -1:].shape == out.shape
            return mean_squared_error(out_pred[:, ::2, -1:], out)

        return training_metric

    def get_eval_metrics(self):
        def eval_metric(out_pred, out, in_data):
            return {}

        return eval_metric


class ODEOperatorICLFinal(Task):
    def __init__(self, noise_variance=0, seed=0, device="cuda", **kwargs):
        super().__init__(name="ode_operator", **kwargs)
        self.device = device
        self.rng = np.random.default_rng(seed=seed)
        self.noise_variance = noise_variance

    def evaluate(self, task_data):
        c_eval = task_data["c_eval"]  # (batch_size, n_points, n_samples)
        B, L, D = c_eval.shape

        # Chebyshev interpolate onto sampled points
        t_eval = self.rng.uniform(-1, 1, B)
        u_cheb = task_data["u_cheb"]
        u_eval = interpolate_cheb(u_cheb, t_eval)  # (batch_size, n_points, n_samples)
        u_eval = u_eval[np.arange(B), :, np.arange(B)]  # (batch_size, n_points)

        if self.noise_variance != 0:
            u_ood = u_eval + self.noise_variance * self.rng.standard_normal(
                u_eval.shape
            )
        else:
            u_ood = u_eval

        # Inputs: [(c, u0, 1, u_ood)]. Last element is the query, (c, u0, 1) is the context.
        # Outputs: [u_ood]
        in_data = torch.zeros((B, L, D + 3)).to(device=self.device)  # (B, L, D+3)
        in_data[:, :, :-3] = torch.tensor(c_eval)
        in_data[:, :, -3] = torch.tensor(task_data["u0"])
        in_data[:, :, -2] = 1
        in_data[:, :-1, -1] = torch.tensor(u_ood)[:, :-1]

        out_data = (
            torch.tensor(u_eval)[:, -1:].to(device=self.device).unsqueeze(-1)
        )  # (B, 1, 1)

        return {
            "in": in_data,  # (B, L, D+3)
            "out": out_data,  # (B, 1, 1)
        }

    def get_metric(self):
        def metric(out_pred, out):
            assert out_pred[:, -1:, -1:].shape == out.shape
            return squared_error(out_pred[:, -1:, -1:], out)

        return metric

    def get_training_metric(self):
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
    n_dims = 21
    n_points = 25
    batch_size = 64
    seed = 0

    # Test ODEOperatorSampler
    sampler = ODEOperatorSampler(
        n_dims,
        c_sampling="equispaced",
        eqn_class=0,
        operator_scale=1,
        u0_scale=1,
        seed=seed,
        device="cuda",
    )

    # Check shapes of task_data
    task_data = sampler.sample(n_points, batch_size, n_dims_truncated=n_dims)
    assert task_data["c_cheb"].shape == (batch_size, n_points, 41)
    assert task_data["c_eval"].shape == (batch_size, n_points, n_dims)
    assert task_data["u0"].shape == (batch_size, n_points)
    assert task_data["u_cheb"].shape == (batch_size, n_points, 41)

    # Test ODEOperatorICL
    task = ODEOperatorICL(
        noise_variance=0,
        seed=seed,
        device="cuda",
    )
    out = task.evaluate(task_data)

    # Check shapes of out
    assert out["in"].shape == (batch_size, 2 * n_points - 1, n_dims + 3)
    assert out["out"].shape == (batch_size, n_points, 1)

    # Test ODEOperatorICLFinal
    task = ODEOperatorICLFinal(
        noise_variance=0,
        seed=seed,
        device="cuda",
    )
    out = task.evaluate(task_data)

    # Check shapes of out
    assert out["in"].shape == (batch_size, n_points, n_dims + 3)
    assert out["out"].shape == (batch_size, 1, 1)

    print("ODEOperatorSampler and ODEOperatorICL test passed")
