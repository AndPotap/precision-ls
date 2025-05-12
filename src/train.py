import math
import os
import uuid

import torch
import wandb
import yaml
from quinine import QuinineArgumentParser
from tqdm import tqdm

from datagen.main import get_data_sampler, get_task_sampler
from models import build_model
from optimizers import (
    compute_current_gradient_stats,
    compute_gradient_stats,
    grad_history_to_tensor,
    gradfilter_ema,
)
from schedulers import AdaptiveStepLR, StepThresholdLR
from train_util import Curriculum, get_device, schema

torch.backends.cudnn.benchmark = True

os.environ["WANDB__SERVICE_WAIT"] = "300"


def compute_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def train_step(model, xs, ys, optimizer, scheduler, loss_func, ema_args=None, out_dict=None):
    optimizer.zero_grad()

    output = model(xs)  # (b l d)
    loss = loss_func(output, ys)
    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
    grad_norm = compute_gradient_norms(model)

    # If grad norm is too high, skip the step
    if grad_norm > 1e3 or math.isnan(grad_norm):
        print(f"Gradient norm too high: {grad_norm}, skipping step")
        return out_dict

    if ema_args is not None:
        ema_grads = ema_args["ema_grads"]
        ema_history = ema_args["ema_history"]
        ema_decay = ema_args["ema_decay"]
        ema_lambda = ema_args["ema_lambda"]
        ema_grads, ema_history = gradfilter_ema(
            model,
            grads=ema_grads,
            grad_history=ema_history,
            alpha=ema_decay,
            lamb=ema_lambda,
        )
    else:
        ema_grads = None
        ema_history = None

    optimizer.step()
    grad_norm = compute_gradient_norms(model)
    grad_norm_dict = {
        "norm": grad_norm,
    }

    scheduler.step()

    out_dict = {
        "loss": loss.detach().item(),
        "output": output.detach(),
        "grad_norm_dict": grad_norm_dict,
        "ema_grads": ema_grads,
        "ema_history": ema_history,
    }

    return out_dict


def train(model, args):
    if args.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.training.learning_rate,
            weight_decay=args.training.weight_decay,
            eps=1e-16,
        )
    elif args.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.training.learning_rate,
            weight_decay=args.training.weight_decay,
            momentum=0.9,
        )
    elif args.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.training.learning_rate,
            weight_decay=args.training.weight_decay,
            eps=1e-16,
        )
    else:
        raise ValueError(f"Invalid optimizer: {args.training.optimizer}")

    if args.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.training.scheduler_steprate,
            gamma=args.training.scheduler_gamma,
        )
    elif args.training.scheduler == "step_threshold":
        scheduler = StepThresholdLR(
            optimizer,
            step_size=args.training.scheduler_steprate,
            gamma=args.training.scheduler_gamma,
            threshold=1e-10,
        )
    elif args.training.scheduler == "adaptive_threshold":
        scheduler = AdaptiveStepLR(
            optimizer,
            step_size=args.training.scheduler_steprate,
            gamma=args.training.scheduler_gamma,
            threshold=1e-10,
            gradient_variance_metric_lambda=args.training.scheduler_ema_lambda,
            gradient_variance_threshold=args.training.scheduler_metric_threshold,
            no_increase_length=1e3,
        )
    else:
        raise ValueError(f"Invalid scheduler: {args.training.scheduler}")

    curriculum = Curriculum(args.training.curriculum)

    if args.training.dtype == "float32":
        dtype = torch.float32
    elif args.training.dtype == "float64":
        dtype = torch.float64
    else:
        raise ValueError(f"Invalid dtype: {args.training.dtype}")
    device = get_device()

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(
        data_name=args.training.data,
        n_dims=n_dims,
        batch_size=bsize,
        **args.training.data_kwargs,
    )
    data_sampler_args = {}
    task_sampler = get_task_sampler(
        task_name=args.training.task,
        n_dims=n_dims,
        batch_size=bsize,
        n_points=curriculum.n_points,
        **args.training.task_kwargs,
    )
    task_sampler_args = {}
    task = task_sampler(**task_sampler_args)

    pbar = tqdm(range(starting_step, args.training.train_steps))

    for i in pbar:
        task_data = data_sampler.sample(
            n_points=curriculum.n_points,
            batch_size=bsize,
            n_dims_truncated=curriculum.n_dims_truncated,
            **data_sampler_args,
        )

        task_out = task.evaluate(task_data)
        xs = task_out["in"].to(dtype=dtype, device=device)
        ys = task_out["out"].to(dtype=dtype, device=device)

        # Initialize loss function
        if i == 0 or args.training.resume_id is not None:
            l1_scale = 1  # Scale L1 norm by loss

            # For EMA over gradients
            ema_grads = None
            ema_history = None
            out_dict = None  # gets overwritten each step

        else:
            # l1_scale = min(training_metrics_task, l1_scale, 1)
            l1_scale = min(l1_scale, 1)

        l1_norm = sum(p.abs().sum() for p in model.parameters())

        def loss_func(output, ys):
            out = task.get_training_metric()(output, ys)
            out = out + args.training.l1_reg * l1_scale * l1_norm
            return out

        # loss_func = (
        #     lambda output, ys: task.get_training_metric()(output, ys)
        #     + args.training.l1_reg * l1_scale * l1_norm
        # )  # L1 regularization

        # Compute mini-batch gradient variance
        if (i % args.wandb.log_every_steps == 0) and not args.test_run:
            # Compute and track gradient stats
            if args.training.track_grad or scheduler.__class__.__name__ == "AdaptiveStepLR":
                # Mini-batch gradient variance
                def sample_data(batch_size):
                    task_data = data_sampler.sample(
                        n_points=curriculum.n_points,
                        batch_size=batch_size,
                        n_dims_truncated=curriculum.n_dims_truncated,
                        **data_sampler_args,
                    )
                    task_out = task.evaluate(task_data)
                    xs = task_out["in"].to(dtype=dtype, device=device)
                    ys = task_out["out"].to(dtype=dtype, device=device)
                    return xs, ys

                # For current model, sample gradients from minibatches and compute statistics
                gradient_stats = compute_current_gradient_stats(
                    model,
                    task.get_training_metric(),
                    n_batches=64,
                    batch_size=bsize,
                    data_sampler=sample_data,
                )

                # Wandb
                wandb.log(gradient_stats, step=i)

                # Update EMA over gradient variance metric within adaptive LR scheduler
                if scheduler.__class__.__name__ == "AdaptiveStepLR":
                    scheduler.update_gradient_variance_metric(gradient_stats["gradient_cosine_mean"])
                    gradient_variance_metric_ema = scheduler.gradient_variance_metric_ema
                    wandb.log(
                        {"gradient_cosine_metric_lrscheduler": gradient_variance_metric_ema},
                        step=i,
                    )

                # Compute gradient statistics for EMA
                if ema_history is not None:
                    gradient_stats_ema = compute_gradient_stats(
                        grad_history_to_tensor(ema_history),
                        prefix="ema",
                    )
                    # Wandb
                    wandb.log(gradient_stats_ema, step=i)

        # Training step
        out_dict = train_step(
            model,
            xs,
            ys,
            optimizer,
            scheduler,
            loss_func,
            ema_args={
                "ema_grads": ema_grads,
                "ema_history": ema_history,
                "ema_decay": args.training.ema_decay,
                "ema_lambda": args.training.ema_lambda,
            },
            out_dict=out_dict,
        )
        rhs = xs[..., [-1]]
        A = xs[..., :-1]
        x_hat = out_dict["output"][..., [-1], :].transpose(-1, -2)
        actual = torch.mean(torch.linalg.norm(A @ x_hat - rhs, ord=2, dim=[-1, -2]))

        loss = out_dict["loss"]
        output = out_dict["output"]
        grad_norms = out_dict["grad_norm_dict"]
        ema_grads = out_dict["ema_grads"]
        ema_history = out_dict["ema_history"]

        # Pointwise losses
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=(0, 2))
        point_wise_tags = list(range(point_wise_loss.shape[0]))

        # Evaluation metrics
        eval_metrics_func = task.get_eval_metrics()
        eval_metrics = eval_metrics_func(output, ys.to(device), xs.to(device))  # dict of metrics
        # Training metrics
        training_metrics_task = task.get_training_metric()(output, ys.to(device))
        training_metrics_reg = l1_norm

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "pointwise/loss": dict(zip(point_wise_tags, point_wise_loss.cpu().numpy())),
                    "least_squares": actual.item(),
                    "task_loss": training_metrics_task,
                    "reg_loss": training_metrics_reg,
                    "grad_norm": grad_norms["norm"],
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "learning_rate": scheduler.get_last_lr()[0],
                },
                step=i,
            )

            # Eval metrics
            for key, value in eval_metrics.items():
                wandb.log({key: value.item()}, step=i)

        curriculum.update()

        text = f"Overall loss:  {loss:.3e} / Task loss: {training_metrics_task:.3e} / Reg loss: {training_metrics_reg:.3e} / "
        text += f"LS: {actual:.3e} / "
        text += f"Grad norm: {grad_norms['norm']:.3e} / Learning rate: {scheduler.get_last_lr()[0]}"
        pbar.set_description(text)

        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
            mode=args.wandb.mode,
        )

    # Set torch dtype
    if args.training.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif args.training.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f"Invalid dtype: {args.training.dtype}")

    model = build_model(args.model)
    print(f"Number of params: {sum(p.numel() for p in model.parameters())}")
    model.to(get_device())
    model.train()

    # Sanity check: assert that each model that requires_grad has a name
    num_unique_namedparams = len(set([name for name, param in model.named_parameters() if param.requires_grad]))
    num_unique_params = len([param for param in model.parameters() if param.requires_grad])
    assert num_unique_namedparams == num_unique_params, (
        f"Number of unique named parameters ({num_unique_namedparams}) does not match number of unique parameters ({num_unique_params})"
    )

    train(model, args)


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    # Seed
    torch.manual_seed(args.training.seed)

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        if args.autoname:
            save_name = f"{args.out_dir}_seqop={args.model.seq_op}_l={args.model.n_layer}_hdim={args.model.n_embd}_mlp={args.model.use_mlps}_mlpupfactor={args.model.mlp_upfactor}_useseqln={args.model.use_seqop_ln}"
            out_dir = os.path.join(save_name, run_id)
            args.wandb.name = save_name.split("/")[-1]
        else:
            out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
