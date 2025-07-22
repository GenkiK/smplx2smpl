from typing import Callable

import numpy as np
import torch
from torch import optim
from tqdm import tqdm


def build_optimizer(parameters: list[torch.Tensor], optim_cfg: dict) -> dict:
    """Creates the optimizer"""
    optim_type = optim_cfg.get("type", "sgd")

    parameters = list(filter(lambda x: x.requires_grad, parameters))

    if optim_type == "adam":
        optimizer = optim.Adam(parameters, **optim_cfg.get("adam", {}))
        create_graph = False
    elif optim_type == "lbfgs" or optim_type == "lbfgsls":
        optimizer = optim.LBFGS(parameters, **optim_cfg.get("lbfgs", {}))
        create_graph = False
    elif optim_type == "rmsprop":
        optimizer = optim.RMSprop(parameters, **optim_cfg.get("rmsprop", {}))
        create_graph = False
    elif optim_type == "sgd":
        optimizer = optim.SGD(parameters, **optim_cfg.get("sgd", {}))
        create_graph = False
    else:
        raise ValueError(f"Optimizer {optim_type} not supported!")
    return {"optimizer": optimizer, "create_graph": create_graph}


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def minimize(
    optimizer: torch.optim,
    closure,
    params: list[torch.Tensor],
    summary_closure: Callable[[], dict[str, float]] | None = None,
    maxiters=100,
    ftol=-1.0,
    gtol=1e-9,
    verbose=True,
    summary_steps=10,
    **kwargs,
):
    """Helper function for running an optimization process
    Args:
        - optimizer: The PyTorch optimizer object
        - closure: The function used to calculate the gradients
        - params: a list containing the parameters that will be optimized
    Keyword arguments:
        - maxiters (100): The maximum number of iterations for the
          optimizer
        - ftol: The tolerance for the relative change in the loss
          function.
          If it is lower than this value, then the process stops
        - gtol: The tolerance for the maximum change in the gradient.
          If the maximum absolute values of the all gradient tensors
          are less than this, then the process will stop.
    """
    prev_loss = None
    for n in tqdm(range(maxiters), desc="Fitting iterations"):
        # for n in range(maxiters):
        loss = optimizer.step(closure)

        if torch.isnan(loss) or torch.isinf(loss):
            break

        if n > 0 and prev_loss is not None and ftol > 0:
            loss_rel_change = rel_change(prev_loss, loss.item())

            if loss_rel_change <= ftol:
                prev_loss = loss.item()
                break

        if all([var.grad.view(-1).abs().max().item() < gtol for var in params if var.grad is not None]) and gtol > 0:
            prev_loss = loss.item()
            break

        if verbose and n % summary_steps == 0:
            print(f"[{n:05d}] Loss: {loss.item():.4f}", flush=True)
            if summary_closure is not None:
                summaries = summary_closure()
                for key, val in summaries.items():
                    print(f"[{n:05d}] {key}: {val:.4f}", flush=True)

        prev_loss = loss.item()

    # Save the final step
    if verbose:
        print(f"[{n + 1:05d}] Loss: {loss.item():.4f}", flush=True)
        if summary_closure is not None:
            summaries = summary_closure()
            for key, val in summaries.items():
                print(f"[{n + 1:05d}] {key}: {val:.4f}", flush=True)

    return prev_loss


def get_variables(
    batch_size: int,
    smpl_model,
    dtype: torch.dtype = torch.float32,
    prev_result_dict: dict[str, torch.Tensor] | None = None,
    fix_betas: bool = False,
) -> dict[str, torch.Tensor]:
    device = next(smpl_model.buffers()).device
    if prev_result_dict is not None:
        var_dict = {
            "transl": prev_result_dict["transl"].detach().clone().to(device),
            "global_orient": prev_result_dict["global_orient"].detach().clone().to(device),
            "body_pose": prev_result_dict["body_pose"].detach().clone().to(device),
            "betas": prev_result_dict["betas"].detach().clone().to(device),
        }
    else:
        var_dict = {
            "transl": torch.zeros([batch_size, 3], device=device, dtype=dtype),
            "global_orient": torch.zeros([batch_size, 1, 3], device=device, dtype=dtype),
            "body_pose": torch.zeros(
                [batch_size, smpl_model.NUM_BODY_JOINTS, 3],
                device=device,
                dtype=dtype,
            ),
            "betas": torch.zeros([batch_size, smpl_model.num_betas], dtype=dtype, device=device),
        }

    # Toggle gradients to True
    for k, v in var_dict.items():
        if fix_betas and k == "betas":
            v.requires_grad_(False)
        else:
            v.requires_grad_(True)

    return var_dict
