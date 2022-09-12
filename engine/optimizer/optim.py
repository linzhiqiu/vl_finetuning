"""optim.py.
"""

import torch

AVAI_OPTIMS = ["adam", "sgd", "adamw"]

def build_optimizer(params_groups, cfg, name, lr, weight_decay):
    """Build optimizer.
    Args:
        name (str): name of optimizer.
        params_groups (list[dict]): list of parameters groups.
        **kwargs: other arguments.
    """
    assert name in AVAI_OPTIMS, f"Optimizer {name} not found; available optimizers = {AVAI_OPTIMS}"
    if name == "sgd":
        return build_sgd_optimizer(params_groups, lr, weight_decay, cfg.OPTIM.MOMENTUM, cfg.OPTIM.SGD_NESTEROV)
    elif name == "adam":
        return build_adam_optimizer(params_groups, lr, weight_decay, betas=(cfg.OPTIM.ADAM_BETA1, cfg.OPTIM.ADAM_BETA2))
    elif name == "adamw":
        return build_adamw_optimizer(params_groups, lr, weight_decay, betas=(cfg.OPTIM.ADAM_BETA1, cfg.OPTIM.ADAM_BETA2))
    else:
        raise ValueError("Unknown optimizer: {}".format(name))
    

def build_sgd_optimizer(params_groups, lr, weight_decay, momentum=0.9, nesterov=False):
    """Build SGD optimizer.
    Args:
        params_groups (list[dict]): list of parameters groups.
        lr (float): learning rate.
        momentum (float): momentum.
        weight_decay (float): weight decay.
        nesterov (bool): whether to use nesterov.
    """
    return torch.optim.SGD(
        params_groups,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )

def build_adam_optimizer(params_groups, lr, weight_decay, betas=(0.9, 0.999)):
    """Build Adam optimizer.
    Args:
        params_groups (list[dict]): list of parameters groups.
        lr (float): learning rate.
        betas (tuple[float]): coefficients used for computing running averages of gradient and its square.
        weight_decay (float): weight decay.
    """
    return torch.optim.Adam(
        params_groups, lr=lr, weight_decay=weight_decay, betas=betas
    )

def build_adamw_optimizer(params_groups, lr, weight_decay, betas=(0.9, 0.999)):
    """Build AdamW optimizer.
    Args:
        params_groups (list[dict]): list of parameters groups.
        lr (float): learning rate.
        betas (tuple[float]): coefficients used for computing running averages of gradient and its square.
        weight_decay (float): weight decay.
    """
    return torch.optim.AdamW(
        params_groups, lr=lr, weight_decay=weight_decay, betas=betas
    )