# 0. Sweep over all experiments
    # 1. For best_val and last_iter model, 
    # We evaluate their performance of
    # a - as-is
    # b - wiseFT (add normalized zero-shot head)
    # c - normalized head
    # d - normalized head + wiseFT
# 1. Sweep over seeds and report the average performance
    # 2. Find best val model and report their test performance for
    # a - best_val's a/b/c/d
    # b - last_iter's a/b/c/d
    # 3. Find best test model for 
    # a - best_val's a/b/c/d
    # b - last_iter's a/b/c/d
from copy import deepcopy
import os, argparse
import pdb
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
from logreg_minibatch import get_hyperparams_str, get_save_dir, get_valid_batch_sizes, validate


# PARTIAL = True
PARTIAL = False
if PARTIAL:
    #### Partial START
    EVAL_DIR = "./partial_results"

    IMAGES = [
        "rn50_layer_1",
        # "vitb16_layer_1",
        "rn50_layer_2",
        # "vitb16_layer_2",
        # # "vitb16_layer_4",
        # "rn50_layer_all",
        # "vitb16_layer_all",
    ]

    TEXTS = [
        "layer_0",
        #   "layer_1",
        #   "layer_all",
    ]

    TEMPLATES = [
        # "classname",
        # "default",
        # "extra",
        "single",
    ]

    VIEWS = [
        "view_10_valview_10_randomcrop",
        "view_1_ccrop",
        # "view_100_rcrop",
        "view_100_valview_100_rcrop",
    ]

    DATASETS = [
        "imagenet",
        # "caltech101",
        # "dtd",
        # "eurosat",
        # "fgvc_aircraft",
        # "food101",
        # "oxford_flowers",
        # "oxford_pets",
        # "stanford_cars",
        # "sun397",
        # "ucf101",
    ]

    CROSS_MODALS = [
        "text_ratio_0.5",
        "text_ratio_0",
    ]

    LOGITS = [
        # "cosine_logit_scale",
        # "cosine",
        "linear",
    ]

    HYPERS = [
        "partial_adamw"
    ]

    ARCHITECTURES = [
        "linear_zeroshot",
    ]

    SEEDS = [
        3,
        2,
        1,
    ]

    SHOTS = [
        # 1,
        # 2,
        # 4,
        8,
        # 16
    ]
    #### Partial END
else:
    # EVAL_DIR = "./logreg_minibatch_results"
    EVAL_DIR = "./debug_results"
    ## first layer
    IMAGES = [
        "rn50_layer_0",
        # "vitb16_layer_0",
        # "rn50_layer_1",
        # "vitb16_layer_1",
        # "rn50_layer_2",
        # "vitb16_layer_2",
        # # "vitb16_layer_4",
        # "rn50_layer_all",
        # "vitb16_layer_all",
    ]

    TEXTS = [
        "layer_0",
        #   "layer_1",
        #   "layer_all",
    ]

    TEMPLATES = [
        # "classname",
        # "default",
        # "extra",
        "single",
    ]

    VIEWS = [
        # "view_10_valview_10_randomcrop",
        "view_1_ccrop",
        # "view_100_rcrop",
        # "view_100_valview_100_rcrop",
    ]

    DATASETS = [
        "imagenet",
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
        "text_ratio_0.2",
        "text_ratio_0.5",
        "text_ratio_0.8",
        # "text_ratio_1",
    ]

    LOGITS = [
        # "cosine_logit_scale",
        # "cosine",
        "linear",
    ]
    
    HYPERS = [
        "sam_1",
        # "sam_best",
        # "sam",
        # "adamw",
        # "sgd",
    ]

    ARCHITECTURES = [
        # "linear",
        # "linear_bias",
        # "linear_zeroshot_bias",
        # "mlp_bias",
        "linear_zeroshot",
        # "mlp",
    ]

    SEEDS = [
        1,
        2,
        3,
    ]

    SHOTS = [
        # 1,
        # 2,
        # 4,
        # 8,
        16
    ]


def setup_cfg(dataset,
              shots,
              image,
              text,
              template,
              view,
              cross_modal,
              architecture,
              logit,
              hyper,
              seed):
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
    
    # 8. From the architecture config file
    cfg.merge_from_file(f"config/architecture/{architecture}.yaml")

    # 9. From the logit config file
    cfg.merge_from_file(f"config/logit/{logit}.yaml")
    
    # 10. From the hyperparams config file
    cfg.merge_from_file(f"config/hyperparams/logreg_minibatch/{hyper}.yaml")

    # 11. Set the seed
    cfg.SEED = seed

    cfg.freeze()

    return cfg


def get_normhead(head):
    if type(head) == torch.nn.Linear:
        head.weight.data = torch.nn.functional.normalize(head.weight.data, dim=1)
    elif type(head) == torch.nn.Sequential:
        assert type(head[-1]) == torch.nn.Linear, f"Invalid head: {head}"
        head[-1].weight.data = torch.nn.functional.normalize(head[-1].weight.data, dim=1)
    return head

def get_wiseft(head, zero_shot_weights, wiseft_ratio=0.5):
    if type(head) == torch.nn.Linear:
        head.weight.data = (1 - wiseft_ratio) * head.weight.data + wiseft_ratio * zero_shot_weights
    elif type(head) == torch.nn.Sequential:
        assert type(head[-1]) == torch.nn.Linear, f"Invalid head: {head}"
        head[-1].weight.data = (1 - wiseft_ratio) * head[-1].weight.data + wiseft_ratio * zero_shot_weights
    return head

def get_eval_heads(head, zero_shot_weights, ratio_list=[0.2, 0.5, 0.8]):
    logit_head = make_logit_head(
        deepcopy(head), False, False, False)

    normhead = get_normhead(deepcopy(head))
    normhead_head = make_logit_head(
        deepcopy(normhead), False, False, False)

    eval_heads = {
        'head': logit_head.cuda().eval(),
        'normhead': normhead_head.cuda().eval(),
    }
    for ratio in ratio_list:
        wiseft = get_wiseft(deepcopy(head), zero_shot_weights, ratio)
        wiseft_head = make_logit_head(
            wiseft, False, False, False)
        eval_heads[f'head_wiseft_{ratio}'] = wiseft_head.cuda().eval()
        normhead_wiseft = get_wiseft(deepcopy(normhead), zero_shot_weights, ratio)
        normhead_wiseft_head = make_logit_head(
            normhead_wiseft, False, False, False)
        eval_heads[f'normhead_wiseft_{ratio}'] = normhead_wiseft_head.cuda().eval()
    return eval_heads

def take_average(all_seed_dict,
                 ALL_LRS,
                 ALL_WDS,
                 ALL_BATCHSIZES,
                 ALL_ITERS,
                 ):
    header = ['lr', 'wd', 'batchsize', 'max_iters', 'early_stop', 'iter_mean', 'iter_std',
              'eval_type', 'val_acc_mean', 'val_acc_std', 'test_acc_mean', 'test_acc_std']
    columns = []
    ALL_SEEDS = list(all_seed_dict.keys())
    ALL_EVAL_TYPES = None
    avg_dict = {}
    std_dict = {}
    for lr in ALL_LRS:
        avg_dict[lr] = {}
        std_dict[lr] = {}
        for wd in ALL_WDS:
            avg_dict[lr][wd] = {}
            std_dict[lr][wd] = {}
            for batchsize in ALL_BATCHSIZES:
                avg_dict[lr][wd][batchsize] = {}
                std_dict[lr][wd][batchsize] = {}
                for iters in ALL_ITERS:
                    avg_dict[lr][wd][batchsize][iters] = {}
                    std_dict[lr][wd][batchsize][iters] = {}
                    for key in ['best_val', 'last_iter']:
                        avg_dict[lr][wd][batchsize][iters][key] = {}
                        std_dict[lr][wd][batchsize][iters][key] = {}
                        for metric in ['val_acc', 'iter']:
                            avg_dict[lr][wd][batchsize][iters][key][metric] = np.mean(
                                [all_seed_dict[seed][lr][wd][batchsize][iters][key][metric] for seed in ALL_SEEDS])
                            std_dict[lr][wd][batchsize][iters][key][metric] = np.std(
                                [all_seed_dict[seed][lr][wd][batchsize][iters][key][metric] for seed in ALL_SEEDS])
                        avg_dict[lr][wd][batchsize][iters][key]['test_accs'] = {}
                        std_dict[lr][wd][batchsize][iters][key]['test_accs'] = {}
                        if ALL_EVAL_TYPES is None:
                            ALL_EVAL_TYPES = list(all_seed_dict[ALL_SEEDS[0]][lr][wd][batchsize][iters][key]['test_accs'].keys())
                        for eval_type in ALL_EVAL_TYPES:
                            avg_dict[lr][wd][batchsize][iters][key]['test_accs'][eval_type] = np.mean(
                                [all_seed_dict[seed][lr][wd][batchsize][iters][key]['test_accs'][eval_type] for seed in ALL_SEEDS])
                            std_dict[lr][wd][batchsize][iters][key]['test_accs'][eval_type] = np.std(
                                [all_seed_dict[seed][lr][wd][batchsize][iters][key]['test_accs'][eval_type] for seed in ALL_SEEDS])
                            columns.append([lr, wd, batchsize, iters, key, 
                                            avg_dict[lr][wd][batchsize][iters][key]['iter'], std_dict[lr][wd][batchsize][iters][key]['iter'],
                                            eval_type,
                                            avg_dict[lr][wd][batchsize][iters][key]['val_acc'], std_dict[lr][wd][batchsize][iters][key]['val_acc'],
                                            avg_dict[lr][wd][batchsize][iters][key]['test_accs'][eval_type], std_dict[lr][wd][batchsize][iters][key]['test_accs'][eval_type]])
    return header, columns

def get_result_dir(eval_dir=EVAL_DIR):
    return eval_dir


def save_csv(header,
             columns,
             result_path,
             shots,
             image,
             text,
             template,
             view,
             cross_modal,
             architecture,
             logit,
             hyper,
             dataset,):
    all_headers = ['dataset', 'shots', 'image', 'text', 'template', 'view', 'cross_modal', 'architecture', 'logit', 'hyper'] + header
    all_columns = [[dataset, shots, image, text, template, view, cross_modal, architecture, logit, hyper] + column for column in columns]
    save_all_csv(all_headers, all_columns, result_path)
    return all_headers, all_columns

def save_all_csv(all_headers, all_columns, result_path):
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(result_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)
        writer.writerows(all_columns)

def main():
    all_columns = []
    all_headers = None
    for template_idx, template in enumerate(TEMPLATES):
        print(f"Template: {template} | {template_idx + 1}/{len(TEMPLATES)}")
        result_dir = get_result_dir()
        all_dataset_dict = {}
        for dataset_idx, dataset in enumerate(DATASETS):
            print(f"Dataset: {dataset} | {dataset_idx + 1}/{len(DATASETS)}")
            all_seed_dict = {}
            for seed in SEEDS:
                cfg = setup_cfg(dataset,
                                1,
                                'rn50_layer_0',
                                'layer_0',
                                template,
                                'view_1_ccrop',
                                'text_ratio_0',
                                'linear',
                                'linear',
                                'sam_1',
                                seed)
                if torch.cuda.is_available() and cfg.USE_CUDA:
                    torch.backends.cudnn.benchmark = True


                text_features_path = get_text_features_path(cfg)
                text_features = torch.load(text_features_path)
                text_dataset = TextTensorDataset(
                    text_features['features'], text_features['labels'], text_features['eot_indices'])

                test_features_path = get_test_features_path(cfg)
                test_features = torch.load(test_features_path)
                test_dataset = TensorDataset(
                    test_features['features'], test_features['labels'])

                checkpoint_dir = os.path.join(result_dir, dataset, seed)
                test_result_dict = {}
                test_result_path = os.path.join(checkpoint_dir, "test_result.pth")
                if os.path.exists(test_result_path):
                    print(f"Already exists: {hyperparams_str} {cur_count}/{experiment_count}")
                    test_result_dict = torch.load(test_result_path)
                else:
                    test_result_dict = {
                        'best_val': {},
                        'last_iter': {},
                    }
                        for key in ['best_val', 'last_iter']:
                            result_dict = torch.load(os.path.join(checkpoint_dir, f"{key}.pth"))
                            test_result_dict[key]['val_acc'] = result_dict['val_acc']
                            test_result_dict[key]['iter'] = result_dict['iter']
                            test_result_dict[key]['test_accs'] = {}

                                                                
                                                if not all_hyper_finished:
                                                    print(f"Seed {seed} not finished!")
                                                    # break
                                                else:
                                                    all_seed_dict[seed] = all_hyper_dict
                                            
                                            if all_seed_finished:
                                                print(f"Dataset {dataset} finished! Taking average...")
                                                all_dataset_dict[dataset] = take_average(all_seed_dict, cfg.OPTIM.LR, cfg.OPTIM.WEIGHT_DECAY, VALID_BATCH_SIZES, cfg.OPTIM.MAX_ITER)
                                                this_headers, this_columns = save_csv(all_dataset_dict[dataset][0], all_dataset_dict[dataset][1],
                                                         os.path.join(result_dir, dataset, "all_results.csv"),
                                                         shots,
                                                         image,
                                                         text,
                                                         template,
                                                         view,
                                                         cross_modal,
                                                         architecture,
                                                         logit,
                                                         hyper,
                                                         dataset,)
                                                if all_headers == None:
                                                    all_headers = this_headers
                                                all_columns = all_columns + this_columns
                                            else:
                                                print(f"Dataset {dataset} not finished!")
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
    print("Done!")

if __name__ == "__main__":
    with torch.no_grad():
        main()