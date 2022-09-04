import os, argparse

from engine.datasets import dataset_classes

from engine.tools.utils import makedirs, save_as_json, set_random_seed
from engine.config import get_cfg_default
from engine.datasets.benchmark import generate_fewshot_dataset


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
        # If the json file does not exist, then create it
        print(f"Few-shot data does not exist at {few_shot_index_file}. Create it.")
        makedirs(os.path.dirname(few_shot_index_file))
        benchmark = dataset_classes[cfg.DATASET.NAME](cfg)
        few_shot_dataset = generate_fewshot_dataset(
            benchmark.train, benchmark.val, num_shots=cfg.DATASET.NUM_SHOTS, max_val_shots=cfg.DATASET.MAX_VAL_SHOTS)
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
    
