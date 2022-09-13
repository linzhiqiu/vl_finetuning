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


def make_classifier_head(cfg, text_dataset):
    assert cfg.ARCHITECTURE.HEAD in AVAI_HEADS
    if cfg.FEATURE.BACKBONE == 'ViT-B/16':
        in_features = 512
    elif cfg.FEATURE.BACKBONE == 'RN50':
        in_features = 1024
    assert text_dataset.input_tensor.shape[1] == in_features

    num_classes = int(text_dataset.label_tensor.max()) + 1

    if cfg.ARCHITECTURE.HEAD == 'linear':
        head = nn.Linear(in_features, num_classes, bias=cfg.ARCHITECTURE.BIAS)
    elif cfg.ARCHITECTURE.HEAD == 'linear_zeroshot':
        head = nn.Linear(in_features, num_classes, bias=cfg.ARCHITECTURE.BIAS)
        head.weight.data = get_zero_shot_weights(
            text_dataset, num_classes, in_features)
    elif cfg.ARCHITECTURE.HEAD == 'mlp':
        head = nn.Sequential(
            nn.Linear(in_features, in_features // 4,
                      bias=cfg.ARCHITECTURE.BIAS),
            nn.ReLU(),
            nn.Linear(in_features // 4, num_classes,
                      bias=cfg.ARCHITECTURE.BIAS),
        )
    else:
        raise ValueError(f"Invalid head: {cfg.ARCHITECTURE.HEAD}")
    return head, num_classes, in_features
