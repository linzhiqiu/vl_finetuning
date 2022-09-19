# 0. Sweep over all experiments
# 1. Sweep over seeds and report the average performance
from copy import deepcopy
import os, argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import csv
from engine.tools.utils import makedirs, set_random_seed, collect_env_info
from engine.config import get_cfg_default
from engine.datasets.utils import TensorDataset, TextTensorDataset
from engine.model.head import make_classifier_head, get_zero_shot_weights
from engine.model.logit import make_logit_head
from features import get_backbone_name, \
                     get_few_shot_setup_name, \
                     get_view_name, \
                     get_image_features_path, \
                     get_text_features_path, \
                     get_test_features_path, \
                     get_image_encoder_dir, \
                     get_text_encoder_dir
from logreg_fullbatch import get_filename, get_save_dir

EVAL_DIR = "./logreg_fullbatch_results"

IMAGES = [
    "rn50_layer_0",
    # "vitb16_layer_0",
]

TEXTS = [
    "layer_0",
]

TEMPLATES = [
    # "classname",
    # "default",
    # "extra",
    "single",
]

VIEWS = [
    "view_1_ccrop",
    "view_10_valview_10_randomcrop",
    # "view_100_rcrop",
    # "view_100_valview_100_rcrop",
]

DATASETS = [
    # "imagenet",
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc_aircraft",
    "food101",
    "oxford_flowers",
    "oxford_pets",
    "stanford_cars",
    "sun397",
    "ucf101",
]

CROSS_MODALS = [
    "text_ratio_0",
    "text_ratio_0.5",
    "text_ratio_1",
]

SHOTS = [
    1,
    2,
    4,
    8,
    16
]


def setup_cfg(dataset,
              shots,
              image,
              text,
              template,
              view,
              cross_modal):
    cfg = get_cfg_default()

    # 1. From the dataset config file
    cfg.merge_from_file(f"config/datasets/{dataset}.yaml")

    # 2. From the few-shot config file
    cfg.merge_from_file(f"config/few_shot/shot_{shots}.yaml")

    # 3. From the image encoder config file
    cfg.merge_from_file(f"config/features/image/{image}.yaml")

    # 4. From the text encoder config file
    cfg.merge_from_file(f"config/features/text/{text}.yaml")
    
    # 5. From the template text config file
    cfg.merge_from_file(f"config/features/template/{template}.yaml")

    # 6. From the augmentation view config file
    cfg.merge_from_file(f"config/features/view/{view}.yaml")
    
    # 7. From the cross-modal config file
    cfg.merge_from_file(f"config/cross_modal/{cross_modal}.yaml")
    
    # 10. From the hyperparams config file
    cfg.merge_from_file(f"config/hyperparams/logreg_fullbatch/default.yaml")

    cfg.freeze()

    return cfg


def take_average(test_acc_lists):
    header = ['test_acc_mean', 'test_acc_std']
    acc_mean = np.mean(test_acc_lists)
    acc_std = np.std(test_acc_lists)
    columns = [[acc_mean, acc_std]]
    return header, columns

def get_result_dir(shots,
                   image,
                   text,
                   template,
                   view,
                   cross_modal,
                   eval_dir=EVAL_DIR):
    return os.path.join(eval_dir,
                        f"shots_{shots}",
                        f"image_{image}_text_{text}_template_{template}",
                        f"view_{view}",
                        f"cross_modal_{cross_modal}")


def save_csv(header,
             columns,
             result_path,
             shots,
             image,
             text,
             template,
             view,
             cross_modal,
             dataset,):
    all_headers = ['dataset', 'shots', 'image', 'text', 'template', 'view', 'cross_modal'] + header
    all_columns = [[dataset, shots, image, text, template, view, cross_modal] + column for column in columns]
    save_all_csv(all_headers, all_columns, result_path)
    return all_headers, all_columns


def save_all_csv(all_headers, all_columns, csv_path):
    result_dir = os.path.dirname(csv_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(csv_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)
        writer.writerows(all_columns)

def main():
    all_columns = []
    all_headers = None
    for shots_idx, shots in enumerate(SHOTS):
        print(f"Shots: {shots} | {shots_idx + 1}/{len(SHOTS)}")
        for image_idx, image in enumerate(IMAGES):
            print(f"Image: {image} | {image_idx + 1}/{len(IMAGES)}")
            for text_idx, text in enumerate(TEXTS):
                print(f"Text: {text} | {text_idx + 1}/{len(TEXTS)}")
                for template_idx, template in enumerate(TEMPLATES):
                    print(f"Template: {template} | {template_idx + 1}/{len(TEMPLATES)}")
                    for view_idx, view in enumerate(VIEWS):
                        print(f"View: {view} | {view_idx + 1}/{len(VIEWS)}")
                        for cross_modal_idx, cross_modal in enumerate(CROSS_MODALS):
                            print(f"Cross-modal: {cross_modal} | {cross_modal_idx + 1}/{len(CROSS_MODALS)}")
                            result_dir = get_result_dir(shots, image, text, template, view, cross_modal)
                            all_dataset_finished = True
                            all_dataset_dict = {}
                            for dataset_idx, dataset in enumerate(DATASETS):
                                print(f"Dataset: {dataset} | {dataset_idx + 1}/{len(DATASETS)}")
                                all_seed_finished = True
                                all_seed_dict = {}
                                cfg = setup_cfg(dataset,
                                                shots,
                                                image,
                                                text,
                                                template,
                                                view,
                                                cross_modal)


                                save_dir = get_save_dir(cfg)
                                # check if experiment has been done
                                save_path = os.path.join(save_dir, f"default_test_accs.npy")
                                if not os.path.exists(save_path):
                                    all_dataset_finished = False
                                    all_seed_finished = False
                                    print(f"Experiment not finished: {save_path}")
                                    continue
                                else:
                                    test_acc_lists = np.load(save_path)
                                    print(f"Dataset {dataset} finished! Taking average...")
                                    all_dataset_dict[dataset] = take_average(test_acc_lists)
                                    csv_path = os.path.join(result_dir, dataset, "all_fullbatch_results.csv")
                                    print(f"Saving to {csv_path}")
                                    this_headers, this_columns = save_csv(all_dataset_dict[dataset][0], all_dataset_dict[dataset][1], csv_path,
                                             shots,
                                             image,
                                             text,
                                             template,
                                             view,
                                             cross_modal,
                                             dataset)
                                    if all_headers == None:
                                        all_headers = this_headers
                                    all_columns = all_columns + this_columns
                                    # break
                            # if all_dataset_finished:
                            #     print(f"Experiment finished! Saving...")
                            #     save_csv(all_dataset_dict, os.path.join(result_dir, "all_dataset_dict.csv"))
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%Y-%H:%M")
    csv_path = os.path.join(EVAL_DIR, f"{dt_string}.csv")
    print(f"Saving to {csv_path}")
    save_all_csv(all_headers, all_columns, csv_path)

if __name__ == "__main__":
    with torch.no_grad():
        main()