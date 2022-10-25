import torch
import torch.nn as nn
import torch.nn.functional as F

AVAI_HEADS = ['linear', 'linear_zeroshot', 'linear_zeroshot_norm',
              'mlp', 'mlp_zeroshot', 'adapter_zeroshot', 'adapter_zeroshot_debug', 'adapter_zeroshot_0.05', 'adapter_zeroshot_0.01']


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.2):
        super(Adapter, self).__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a = self.fc(x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        return x

def get_zero_shot_weights(text_dataset, num_classes, in_features):
    weights = torch.zeros(num_classes, in_features)
    count = torch.zeros(num_classes)
    for i in range(len(text_dataset)):
        label = text_dataset.label_tensor[i]
        weights[label] += text_dataset.input_tensor[i]
        count[label] += 1
    weights /= count.unsqueeze(1)
    return weights



def make_classifier_head(head_type, backbone_type, bias, text_dataset):
    assert head_type in AVAI_HEADS
    if backbone_type == 'ViT-B/16':
        in_features = 512
    elif backbone_type == 'RN50':
        in_features = 1024
    assert text_dataset.input_tensor.shape[1] == in_features

    num_classes = int(text_dataset.label_tensor.max()) + 1

    if head_type == 'linear':
        head = nn.Linear(in_features, num_classes, bias=bias)
    elif head_type == 'linear_zeroshot':
        head = nn.Linear(in_features, num_classes, bias=bias)
        head.weight.data = get_zero_shot_weights(
            text_dataset, num_classes, in_features)
    elif head_type == 'linear_zeroshot_norm':
        head = nn.Linear(in_features, num_classes, bias=bias)
        head.weight.data = get_zero_shot_weights(
            text_dataset, num_classes, in_features)
        head.weight.data = F.normalize(head.weight.data, dim=1)
    # elif head_type == 'tip_adapter_zeroshot_norm':
    #     head = nn.Linear(in_features, num_classes, bias=bias)
    #     head.weight.data = get_zero_shot_weights_norm(
    #         text_dataset, num_classes, in_features)
    #     head.weight.data = F.normalize(head.weight.data, dim=1)
    elif head_type == 'mlp':
        head = nn.Sequential(
            nn.Linear(in_features, in_features // 4,
                      bias=bias),
            nn.ReLU(),
            nn.Linear(in_features // 4, num_classes,
                      bias=bias),
        )
    elif head_type == 'mlp_zeroshot':
        linear_head = nn.Linear(in_features, num_classes, bias=bias)
        linear_head.weight.data = get_zero_shot_weights(
            text_dataset, num_classes, in_features)
        linear_head.weight.data = F.normalize(linear_head.weight.data, dim=1)
        head = nn.Sequential(
            nn.Linear(in_features, in_features // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, in_features, bias=False),
            nn.ReLU(inplace=True),
            linear_head
        )
    elif head_type == 'adapter_zeroshot':
        linear_head = nn.Linear(in_features, num_classes, bias=bias)
        linear_head.weight.data = get_zero_shot_weights(
            text_dataset, num_classes, in_features)
        linear_head.weight.data = F.normalize(linear_head.weight.data, dim=1)
        # adapter = Adapter(in_features, residual_ratio=0.2)
        adapter = Adapter(in_features, residual_ratio=0.2)
        head = nn.Sequential(
            adapter,
            linear_head
        )
    elif head_type == 'adapter_zeroshot_debug':
        linear_head = nn.Linear(in_features, num_classes, bias=bias)
        linear_head.weight.data = get_zero_shot_weights(
            text_dataset, num_classes, in_features)
        linear_head.weight.data = F.normalize(linear_head.weight.data, dim=1)
        adapter = Adapter(in_features, residual_ratio=0.)
        head = nn.Sequential(
            adapter,
            linear_head
        )
    elif head_type == 'adapter_zeroshot_0.05':
        linear_head = nn.Linear(in_features, num_classes, bias=bias)
        linear_head.weight.data = get_zero_shot_weights(
            text_dataset, num_classes, in_features)
        linear_head.weight.data = F.normalize(linear_head.weight.data, dim=1)
        adapter = Adapter(in_features, residual_ratio=0.05)
        head = nn.Sequential(
            adapter,
            linear_head
        )
    elif head_type == 'adapter_zeroshot_0.01':
        linear_head = nn.Linear(in_features, num_classes, bias=bias)
        linear_head.weight.data = get_zero_shot_weights(
            text_dataset, num_classes, in_features)
        linear_head.weight.data = F.normalize(linear_head.weight.data, dim=1)
        adapter = Adapter(in_features, residual_ratio=0.01)
        head = nn.Sequential(
            adapter,
            linear_head
        )
    else:
        raise ValueError(f"Invalid head: {head_type}")
    return head, num_classes, in_features

