import torch
import torch.nn as nn
import torch.nn.functional as F


def get_head_norm_func(head):
    if type(head) == nn.Linear:
        def head_norm_func(head):
            return torch.norm(head.weight, dim=1)
    elif type(head) == nn.Sequential:
        assert type(head[-1]) == nn.Linear, f"Invalid head: {head}"
        def head_norm_func(head):
            return torch.norm(head[-1].weight, dim=1)
    return head_norm_func

class LogitHead(nn.Module):
    def __init__(self, head, feature_norm=False, head_norm=False, logit_scale=None):
        super().__init__()
        self.head = head
        self.feature_norm = feature_norm
        self.head_norm = head_norm
        if self.head_norm:
            self.head_norm_func = get_head_norm_func(self.head)
        self.logit_scale = logit_scale

    def forward(self, x):
        if self.feature_norm:
            x = F.normalize(x, dim=1)
        if self.head_norm:
            head_norms = self.head_norm_func(self.head)
        x = self.head(x)
        if self.head_norm:
            x = x / head_norms
        if self.logit_scale:
            x = x * self.logit_scale
        return x


def make_logit_head(cfg, head, logit_scale=100.):
    return LogitHead(
        head,
        cfg.LOGIT.FEATURE_NORM,
        cfg.LOGIT.HEAD_NORM,
        logit_scale if cfg.LOGIT.USE_LOGIT_SCALE else None,
    )