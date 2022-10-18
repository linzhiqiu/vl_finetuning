import os
import pdb
import torch
import torchvision
from torchvision.datasets.folder import default_loader
from engine.datasets import dataset_classes
from engine.tools.utils import load_json


class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor, eot_indices):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    def __len__(self):
        return self.input_tensor.size(0)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor.size(0)


class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        img = self.transform(default_loader(item['impath']))

        output = {
            "img": img,
            "label": item['label'],
            "classname": item['classname'],
            "impath": item['impath'],
        }

        return output


def _check_few_shot_dataset_indices(few_shot_dataset, benchmark):
    # Check if the few-shot dataset has correct indices
    for split in ['train', 'val']:
        split_benchmark = getattr(benchmark, split)
        for idx, benchmark_idx in enumerate(few_shot_dataset[split]['indices']):
            assert split_benchmark[benchmark_idx]['label'] == few_shot_dataset[split]['data'][idx]['label']
            assert split_benchmark[benchmark_idx]['impath'] == few_shot_dataset[split]['data'][idx]['impath']


def get_few_shot_benchmark(cfg):
    # Check if the dataset is supported
    assert cfg.DATASET.NAME in dataset_classes
    few_shot_index_file = os.path.join(
        cfg.FEW_SHOT_DIR, cfg.DATASET.NAME, f"shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.json")
    assert os.path.exists(few_shot_index_file), f"Few-shot data does not exist at {few_shot_index_file}."
    benchmark = dataset_classes[cfg.DATASET.NAME](cfg)
    few_shot_dataset = load_json(few_shot_index_file)
    _check_few_shot_dataset_indices(few_shot_dataset, benchmark)
    return {
        'train': few_shot_dataset['train']['data'],
        'val': few_shot_dataset['val']['data'],
        'test': benchmark.test,
        'lab2cname': benchmark.lab2cname,
        'classnames': benchmark.classnames,
    }


class TestDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        img = self.transform(default_loader(item['impath']))

        return img, item['label']

def get_testset(cfg, transform):
    if cfg.DATASET.NAME in dataset_classes:
        benchmark = dataset_classes[cfg.DATASET.NAME](cfg)
        return TestDatasetWrapper(benchmark.test, transform)
    elif cfg.DATASET.NAME == 'food101_test':
        return torchvision.datasets.Food101(root='/data3/zhiqiul/datasets', split='test', transform=transform, download=True)
    elif cfg.DATASET.NAME == 'dtd_test':
        return torchvision.datasets.DTD(root='/data3/zhiqiul/datasets', split='test', transform=transform, download=True)
    else:
        raise NotImplementedError()

def get_label_map(cfg, dataset_name):
    if dataset_name in ['imagenet_a', 'imagenet_r']:
        benchmark = dataset_classes[dataset_name](cfg)
        return benchmark.label_map
    else:
        return None