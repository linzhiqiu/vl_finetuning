import numpy as np
import os
import torch
from sklearn.linear_model import LogisticRegression
import argparse

from engine.tools.utils import makedirs
from engine.config import get_cfg_default
from features import get_backbone_name, \
                     get_view_name, \
                     get_image_features_path, \
                     get_text_features_path, \
                     get_test_features_path


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def assert_last_layer(cfg):
    assert cfg.TEXT_FEATURE.LAYER_IDX == 0
    assert cfg.FEATURE.LAYER_IDX == 0


def get_benchmark_name(cfg):
    benchmark_name = "-".join([cfg.DATASET.NAME,
                              f"shot_{cfg.DATASET.NUM_SHOTS}"])
    return benchmark_name


def get_text_feature_name(cfg):
    text_feature_name = cfg.TEXT_FEATURE.TEMPLATE
    return text_feature_name


def get_image_feature_name(cfg):
    image_feature_name = get_view_name(cfg)
    return image_feature_name


def get_cross_modal_name(cfg):
    text_feature_name = get_text_feature_name(cfg)
    image_feature_name = get_image_feature_name(cfg)
    if cfg.MODALITY.TEXT_BATCH_RATIO == 0:
        feature_name = image_feature_name
    elif cfg.MODALITY.TEXT_BATCH_RATIO == 1:
        feature_name = text_feature_name
    elif cfg.MODALITY.TEXT_BATCH_RATIO == 0.5:
        feature_name = f"{text_feature_name}-{image_feature_name}-both"
    else:
        raise ValueError("Invalid text_batch_ratio")
    return os.path.join(
        get_backbone_name(cfg),
        feature_name
    )


def get_save_dir(cfg):
    save_dir = os.path.join(
        cfg.LOGREG_FULLBATCH_DIR,
        get_benchmark_name(cfg),
        get_cross_modal_name(cfg),
    )
    return save_dir


def logistic_regression_fullbatch(cfg, save_path):
    test_acc_step_list = np.zeros(
        [len(cfg.LOGREG_FULLBATCH.SEEDS), cfg.LOGREG_FULLBATCH.NUM_STEP])
    for seed_idx, seed in enumerate(cfg.LOGREG_FULLBATCH.SEEDS):
        cfg.SEED = seed
        np.random.seed(seed)
        print(f"-- Seed: {seed} --------------------------------------------------------------")
        text_features_path = get_text_features_path(cfg)
        text_features = torch.load(text_features_path)
        text_feature = text_features['features'].numpy()
        text_label = text_features['labels'].numpy()

        image_features_path = get_image_features_path(cfg)
        image_features = torch.load(image_features_path)
        image_train_feature = image_features['train']['features'].numpy()
        image_train_label = image_features['train']['labels'].numpy()
        val_feature = image_features['val']['features'].numpy()
        val_label = image_features['val']['labels'].numpy()

        test_features_path = get_test_features_path(cfg)
        test_features = torch.load(test_features_path)
        test_feature = test_features['features'].numpy()
        test_label = test_features['labels'].numpy()

        if cfg.MODALITY.TEXT_BATCH_RATIO == 0:
            # Image only
            train_feature = image_train_feature
            train_label = image_train_label
        elif cfg.MODALITY.TEXT_BATCH_RATIO == 1:
            # Text only
            train_feature = text_feature
            train_label = text_label
        elif cfg.MODALITY.TEXT_BATCH_RATIO == 0.5:
            # Both
            train_feature = np.concatenate([image_train_feature, text_feature], axis=0)
            train_label = np.concatenate([image_train_label, text_label], axis=0)
        else:
            raise ValueError("Invalid text_batch_ratio")
        

        # search initialization
        search_list = cfg.LOGREG_FULLBATCH.SEARCH_LIST
        acc_list = []
        for c_weight in search_list:
            clf = LogisticRegression(
                solver="lbfgs", max_iter=cfg.LOGREG_FULLBATCH.MAX_ITER, penalty="l2", C=c_weight).fit(train_feature, train_label)
            pred = clf.predict(val_feature)
            acc_val = sum(pred == val_label) / len(val_label)
            acc_list.append(acc_val)

        print(acc_list, flush=True)

        # binary search
        peak_idx = np.argmax(acc_list)
        c_peak = search_list[peak_idx]
        c_left, c_right = 1e-1 * c_peak, 1e1 * c_peak

        def binary_search(c_left, c_right):
            clf_left = LogisticRegression(
                solver="lbfgs", max_iter=cfg.LOGREG_FULLBATCH.MAX_ITER, penalty="l2", C=c_left).fit(train_feature, train_label)
            pred_left = clf_left.predict(val_feature)
            acc_left = sum(pred_left == val_label) / len(val_label)
            print("Val accuracy (Left): {:.2f}".format(100 * acc_left), flush=True)

            clf_right = LogisticRegression(
                solver="lbfgs", max_iter=cfg.LOGREG_FULLBATCH.MAX_ITER, penalty="l2", C=c_right).fit(train_feature, train_label)
            pred_right = clf_right.predict(val_feature)
            acc_right = sum(pred_right == val_label) / len(val_label)
            print("Val accuracy (Right): {:.2f}".format(100 * acc_right), flush=True)

            # find maximum and update ranges
            if acc_left < acc_right:
                c_final = c_right
                clf_final = clf_right
                # range for the next step
                c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_right = np.log10(c_right)
            else:
                c_final = c_left
                clf_final = clf_left
                # range for the next step
                c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_left = np.log10(c_left)

            pred = clf_final.predict(test_feature)
            test_acc = 100 * sum(pred == test_label) / len(pred)
            print("Test Accuracy: {:.2f}".format(test_acc), flush=True)
            test_acc_step_list[seed_idx - 1, step] = test_acc

            saveline = "{}, seed_idx {}, {} shot, weight {}, test_acc {:.2f}\n".format(cfg.DATASET.NAME, seed_idx, cfg.DATASET.NUM_SHOTS, c_final, test_acc)
            with open(save_path, "a+") as writer:
                writer.write(saveline)
            return (
                np.power(10, c_left),
                np.power(10, c_right),
            )

        for step in range(cfg.LOGREG_FULLBATCH.NUM_STEP):
            print(
                f"{cfg.DATASET.NAME}, {cfg.DATASET.NUM_SHOTS} Shot, Round {step}: {c_left}/{c_right}",
                flush=True,
            )
            c_left, c_right = binary_search(c_left, c_right)
    # save results of last step
    test_acc_list = test_acc_step_list[:, -1]
    acc_mean = np.mean(test_acc_list)
    acc_std = np.std(test_acc_list)
    save_line = "{}, {} Shot, Test acc stat: {:.2f} ({:.2f})\n".format(
        cfg.DATASET.NAME, cfg.DATASET.NUM_SHOTS, acc_mean, acc_std)
    print(save_line, flush=True)
    with open(save_path, "a+") as writer:
        writer.write(save_line)


def setup_cfg(args):
    cfg = get_cfg_default()

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the few-shot config file
    if args.few_shot_config_file:
        cfg.merge_from_file(args.few_shot_config_file)

    # 3. From the image encoder config file
    if args.image_encoder_config_file:
        cfg.merge_from_file(args.image_encoder_config_file)

    # 4. From the text encoder config file
    if args.text_encoder_config_file:
        cfg.merge_from_file(args.text_encoder_config_file)

    # 5. From the template text config file
    if args.template_config_file:
        cfg.merge_from_file(args.template_config_file)

    # 6. From the augmentation view config file
    if args.view_config_file:
        cfg.merge_from_file(args.view_config_file)

    # 7. From the cross-modal config file
    if args.cross_modal_config_file:
        cfg.merge_from_file(args.cross_modal_config_file)

    # 8. From the hyperparams config file
    if args.hyperparams_config_file:
        cfg.merge_from_file(args.hyperparams_config_file)

    # cfg.freeze() # Not freezing in order to sweep the seeds

    return cfg


def main(args):
    cfg = setup_cfg(args)
    assert_last_layer(cfg)

    save_dir = get_save_dir(cfg)
    makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{get_filename(args.hyperparams_config_file)}_log.txt")
    logistic_regression_fullbatch(cfg, save_path)


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
        "--image-encoder-config-file",
        type=str,
        default="",
        help="path to config file for image feature encoder",
    )
    parser.add_argument(
        "--text-encoder-config-file",
        type=str,
        default="",
        help="path to config file for text feature encoder",
    )
    parser.add_argument(
        "--template-config-file",
        type=str,
        default="",
        help="path to config file for text template",
    )
    parser.add_argument(
        "--view-config-file",
        type=str,
        default="",
        help="path to config file for image augmentation views",
    )
    parser.add_argument(
        "--cross-modal-config-file",
        type=str,
        default="",
        help="path to config file for cross-modal training",
    )
    parser.add_argument(
        "--hyperparams-config-file",
        type=str,
        default="",
        help="path to config file for hyperparams",
    )
    args = parser.parse_args()
    main(args)
