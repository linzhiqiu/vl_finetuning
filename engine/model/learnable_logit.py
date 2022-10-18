import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_head_norm_func(head):
    if type(head) == nn.Linear:
        def head_norm_func(head):
            return torch.norm(head.weight, dim=1)
    elif type(head) == nn.Sequential:
        assert type(head[-1]) == nn.Linear, f"Invalid head: {head}"
        def head_norm_func(head):
            return torch.norm(head[-1].weight, dim=1)
    return head_norm_func

class LearnableLogitHead(nn.Module):
    def __init__(self, head, feature_norm=False, head_norm=False, logit_scale=None, learn_logit_scale=False, init_learn_logit_scale=np.log(1 / 0.07)):
        super().__init__()
        self.head = head
        self.feature_norm = feature_norm
        self.head_norm = head_norm
        if self.head_norm:
            self.head_norm_func = get_head_norm_func(self.head)
        if logit_scale is None and learn_logit_scale:
            print(f"LearnableLogitHead: logit_scale is None, setting to {init_learn_logit_scale} and learn it.")
            self.logit_scale = torch.nn.Parameter(torch.ones([]) * init_learn_logit_scale)
        elif logit_scale is not None:
            print(f"LearnableLogitHead: logit_scale is not None, setting to logit_scale {logit_scale}.")
            self.logit_scale = torch.FloatTensor([logit_scale]).cuda()
        else:
            print(f"LearnableLogitHead: logit_scale is None and learn_logit_scale is False, setting to None.")
            self.logit_scale = None

    def forward(self, x):
        if self.feature_norm:
            x = F.normalize(x, dim=1)
        if self.head_norm:
            head_norms = self.head_norm_func(self.head)
        x = self.head(x)
        if self.head_norm:
            x = x / head_norms
        if self.logit_scale:
            x = x * self.logit_scale.exp()
        return x


def make_logit_head(head, feature_norm, head_norm, use_logit_scale, logit_scale=4.6052, learn_logit_scale=False, init_learn_logit_scale=np.log(1 / 0.07)):
    assert not (use_logit_scale and learn_logit_scale), "Cannot use both logit_scale and learn_logit_scale"
    return LearnableLogitHead(
        head,
        feature_norm=feature_norm,
        head_norm=head_norm,
        logit_scale=logit_scale if use_logit_scale else None,
        learn_logit_scale=learn_logit_scale if learn_logit_scale else None,
        init_learn_logit_scale=init_learn_logit_scale,
    )