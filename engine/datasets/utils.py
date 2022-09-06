import os
import torch
from torchvision.datasets.folder import default_loader
from engine.datasets import dataset_classes
from engine.tools.utils import load_json

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