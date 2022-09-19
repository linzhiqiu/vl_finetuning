import torch
import torch.nn as nn

AVAI_HEADS = ['linear', 'linear_zeroshot', 'mlp']


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
    elif head_type == 'mlp':
        head = nn.Sequential(
            nn.Linear(in_features, in_features // 4,
                      bias=bias),
            nn.ReLU(),
            nn.Linear(in_features // 4, num_classes,
                      bias=bias),
        )
    else:
        raise ValueError(f"Invalid head: {head_type}")
    return head, num_classes, in_features

