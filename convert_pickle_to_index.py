import os, argparse

import pickle
from engine.datasets import dataset_classes

from engine.tools.utils import makedirs, save_as_json, set_random_seed
from engine.config import get_cfg_default

def _item_equal(a, b):
    # a is a dict
    # b is a dassl Datum
    if a['impath'] == b.impath and a['label'] == b.label and a['classname'] == b.classname:
        return True
    return False

def _find_index(lst, item):
    for i, x in enumerate(lst):
        if _item_equal(x, item):
            return i
    raise ValueError(f"{item} not found in {lst}")

def _convert(few_shot_pickle, data_source):
    few_shot_dataset = {
        'data': [],
        'indices': [],
    }
    for i, datum in enumerate(few_shot_pickle):
        item = {'impath': datum.impath, 'label': datum.label, 'classname': datum.classname}
        few_shot_dataset['data'].append(item)
        few_shot_dataset['indices'].append(_find_index(data_source, datum))
    assert len(set(few_shot_dataset['indices'])) == len(few_shot_dataset['indices'])
    assert len(few_shot_dataset['data']) == len(few_shot_dataset['indices'])
    assert len(few_shot_dataset['data']) == len(few_shot_pickle)
    return few_shot_dataset

def _convert_pickle_to_json(benchmark, few_shot_train_pickle, few_shot_val_pickle):
    few_shot_train = _convert(few_shot_train_pickle, benchmark.train)
    few_shot_val = _convert(few_shot_val_pickle, benchmark.val)
    few_shot_dataset = {
        'train': few_shot_train,
        'val': few_shot_val,
    }
    return few_shot_dataset

def setup_cfg(args):
    cfg = get_cfg_default()

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the few-shot config file
    if args.few_shot_config_file:
        cfg.merge_from_file(args.few_shot_config_file)

    # 3. Add configs from input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    # Check if the dataset is supported
    assert cfg.DATASET.NAME in dataset_classes
    few_shot_index_file = os.path.join(
        cfg.FEW_SHOT_DIR, cfg.DATASET.NAME, f"shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.json")
    if os.path.exists(few_shot_index_file):
        # If the json file exists, then load it
        print(f"Few-shot data exists at {few_shot_index_file}. Skip.")
    else:
        # If the json file does not exist, then create it from pickle
        print(f"Few-shot data does not exist at {few_shot_index_file}. Create it.")
        makedirs(os.path.dirname(few_shot_index_file))
        benchmark = dataset_classes[cfg.DATASET.NAME](cfg)
        # few_shot_dataset = generate_fewshot_dataset(
        #     benchmark.train, benchmark.val, num_shots=cfg.DATASET.NUM_SHOTS, max_val_shots=cfg.DATASET.MAX_VAL_SHOTS)
        # save_as_json(few_shot_dataset, few_shot_index_file)
        split_fewshot_dir = os.path.join(benchmark.dataset_dir, "split_fewshot")
        assert os.path.exists(split_fewshot_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        seed = cfg.SEED
        preprocessed = os.path.join(
            split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

        assert os.path.exists(preprocessed)
        print(
            f"Loading preprocessed few-shot data from {preprocessed}")
        with open(preprocessed, "rb") as file:
            data = pickle.load(file)
            train, val = data["train"], data["val"]
        
        few_shot_dataset = _convert_pickle_to_json(benchmark, train, val)
        save_as_json(few_shot_dataset, few_shot_index_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "--few-shot-config-file",
        type=str,
        default="",
        help="path to config file for few-shot setup",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
    
