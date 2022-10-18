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
import random
torch.set_num_threads(4)
from torch.utils.data import DataLoader
import numpy as np
import csv
from engine.tools.utils import makedirs, set_random_seed, collect_env_info
from engine.config import get_cfg_default
from engine.datasets.utils import TensorDataset, TextTensorDataset
from engine.model.head import make_classifier_head, get_zero_shot_weights
from engine.model.learnable_logit import make_logit_head
from features import get_backbone_name, \
                     get_few_shot_setup_name, \
                     get_view_name, \
                     get_image_features_path, \
                     get_text_features_path, \
                     get_test_features_path, \
                     get_image_encoder_dir, \
                     get_text_encoder_dir
from final_logit_ablation import get_hyperparams_str, get_save_dir, get_valid_batch_sizes, validate


# EVAL_MODE = 'partial'
EVAL_MODE = 'linear'
if EVAL_MODE == 'partial':
    #### Partial START
    EVAL_DIR = "./final_logit_ablation_results_partial"

    IMAGES = [
        "rn50_layer_1",
    ]

    TEXTS = [
        "layer_0",
    ]

    TEMPLATES = [
        "single",
    ]

    VIEWS = [
        # "view_10_valview_10_randomcrop",
        "view_1_ccrop",
        # "view_100_valview_100_rcrop",
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
        "normtext_ratio_0.5",
        # "text_ratio_0.5",
        "text_ratio_0",
    ]

    LOGITS = [
        "fnorm_False_hnorm_False_logit_Fixed",
        "fnorm_False_hnorm_True_logit_Fixed",
        "fnorm_True_hnorm_False_logit_Fixed",
        "fnorm_True_hnorm_True_logit_Fixed",
        "fnorm_False_hnorm_False_logit_Learn",
        "fnorm_False_hnorm_True_logit_Learn",
        "fnorm_True_hnorm_False_logit_Learn",
        "fnorm_True_hnorm_True_logit_Learn",
    ]

    HYPERS = [
        "partial_adamw_fast"
    ]

    ARCHITECTURES = [
        "linear_zeroshot",
    ]

    SEEDS = [
        1,
        2,
        # 3,
    ]

    SHOTS = [
        1,
        # 2,
        # 4,
        # 8,
        # 16
    ]
    #### Partial END
elif EVAL_MODE == 'linear':
    EVAL_DIR = "./final_logit_ablation_results_linear"
    ## first layer
    IMAGES = ["rn50_layer_0"]

    TEXTS = ["layer_0"]

    TEMPLATES = ["single"]

    VIEWS = [
        "view_1_ccrop",
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
        "text_ratio_0",
        "normtext_ratio_0.5",
        "normtext_ratio_0.2",
        "normtext_ratio_0.8",
    ]
    LOGITS = [
        "fnorm_False_hnorm_False_logit_Fixed",
        "fnorm_False_hnorm_True_logit_Fixed",
        "fnorm_True_hnorm_False_logit_Fixed",
        "fnorm_True_hnorm_True_logit_Fixed",
        "fnorm_False_hnorm_False_logit_Learn",
        "fnorm_False_hnorm_True_logit_Learn",
        "fnorm_True_hnorm_False_logit_Learn",
        "fnorm_True_hnorm_True_logit_Learn",
    ]
    HYPERS = [
        "adamw_2",
    ]

    ARCHITECTURES = [
        "linear_zeroshot",
    ]

    SEEDS = [
        1,
        2,
        3,
    ]

    SHOTS = [
        1,
        2,
        4,
        8,
        16
    ]

if True:
    random.shuffle(LOGITS)
    random.shuffle(SEEDS)
    random.shuffle(SHOTS)


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
    cfg.merge_from_file(f"config/final_logit/{logit}.yaml")
    
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
        # 'normhead': normhead_head.cuda().eval(),
    }
    for ratio in ratio_list:
        wiseft = get_wiseft(deepcopy(head), zero_shot_weights, ratio)
        wiseft_head = make_logit_head(
            wiseft, False, False, False)
        eval_heads[f'head_wiseft_{ratio}'] = wiseft_head.cuda().eval()
        # normhead_wiseft = get_wiseft(deepcopy(normhead), zero_shot_weights, ratio)
        # normhead_wiseft_head = make_logit_head(
        #     normhead_wiseft, False, False, False)
        # eval_heads[f'normhead_wiseft_{ratio}'] = normhead_wiseft_head.cuda().eval()
    return eval_heads

def take_average(all_seed_dict,
                 ALL_LRS,
                 ALL_WDS,
                 ALL_BATCHSIZES,
                 ALL_ITERS,
                 ):
    header = ['lr', 'wd', 'batchsize', 'max_iters', 'early_stop', 'iter_mean', 'iter_std', 'logit_scale_mean', 'logit_scale_std',
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
                        
                        if all_seed_dict[ALL_SEEDS[0]][lr][wd][batchsize][iters][key]['logit_scale'] is not None:
                            avg_dict[lr][wd][batchsize][iters][key]['logit_scale'] = np.mean(
                                [all_seed_dict[seed][lr][wd][batchsize][iters][key]['logit_scale'] for seed in ALL_SEEDS])
                            std_dict[lr][wd][batchsize][iters][key]['logit_scale'] = np.std(
                                [all_seed_dict[seed][lr][wd][batchsize][iters][key]['logit_scale'] for seed in ALL_SEEDS])
                        else:
                            avg_dict[lr][wd][batchsize][iters][key]['logit_scale'] = None
                            std_dict[lr][wd][batchsize][iters][key]['logit_scale'] = None
                        
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
                                            avg_dict[lr][wd][batchsize][iters][key]['logit_scale'], std_dict[lr][wd][batchsize][iters][key]['logit_scale'],
                                            eval_type,
                                            avg_dict[lr][wd][batchsize][iters][key]['val_acc'], std_dict[lr][wd][batchsize][iters][key]['val_acc'],
                                            avg_dict[lr][wd][batchsize][iters][key]['test_accs'][eval_type], std_dict[lr][wd][batchsize][iters][key]['test_accs'][eval_type]])
    return header, columns

def get_result_dir(shots,
                   image,
                   text,
                   template,
                   view,
                   cross_modal,
                   architecture,
                   logit,
                   hyper,
                   eval_dir=EVAL_DIR):
    return os.path.join(eval_dir,
                        f"shots_{shots}",
                        f"image_{image}_text_{text}_template_{template}",
                        f"view_{view}",
                        f"cross_modal_{cross_modal}",
                        f"architecture_{architecture}_logit_{logit}",
                        f"hyper_{hyper}")


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
                            for architecture_idx, architecture in enumerate(ARCHITECTURES):
                                print(f"Architecture: {architecture} | {architecture_idx + 1}/{len(ARCHITECTURES)}")
                                for logit_idx, logit in enumerate(LOGITS):
                                    print(f"Logit: {logit} | {logit_idx + 1}/{len(LOGITS)}")
                                    for hyper_idx, hyper in enumerate(HYPERS):
                                        print(f"Hyper: {hyper} | {hyper_idx + 1}/{len(HYPERS)}")
                                        result_dir = get_result_dir(shots, image, text, template, view, cross_modal, architecture, logit, hyper)
                                        all_dataset_finished = True
                                        all_dataset_dict = {}
                                        for dataset_idx, dataset in enumerate(DATASETS):
                                            print(f"Dataset: {dataset} | {dataset_idx + 1}/{len(DATASETS)}")
                                            all_seed_finished = True
                                            all_seed_dict = {}
                                            for seed in SEEDS:
                                                cfg = setup_cfg(dataset,
                                                                shots,
                                                                image,
                                                                text,
                                                                template,
                                                                view,
                                                                cross_modal,
                                                                architecture,
                                                                logit,
                                                                hyper,
                                                                seed)
                                                if torch.cuda.is_available() and cfg.USE_CUDA:
                                                    torch.backends.cudnn.benchmark = True

                                                image_encoder_dir = get_image_encoder_dir(cfg)
                                                image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")
                                                assert os.path.exists(image_encoder_path), image_encoder_path

                                                text_encoder_dir = get_text_encoder_dir(cfg)
                                                text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
                                                assert os.path.exists(text_encoder_path), text_encoder_path

                                                text_features_path = get_text_features_path(cfg)
                                                text_features = torch.load(text_features_path)
                                                text_dataset = TextTensorDataset(
                                                    text_features['features'], text_features['labels'], text_features['eot_indices'])

                                                test_features_path = get_test_features_path(cfg)
                                                test_features = torch.load(test_features_path)
                                                test_dataset = TensorDataset(
                                                    test_features['features'], test_features['labels'])

                                                save_dir = get_save_dir(cfg)
                                                image_features_path = get_image_features_path(cfg)
                                                image_features = range(torch.load(image_features_path)['train']['features'].shape[0])
                                                VALID_BATCH_SIZES = get_valid_batch_sizes(
                                                    cfg, text_dataset, image_features)
                                                def get_experiment_count(cfg):
                                                    count = 1
                                                    count *= len(cfg.OPTIM.LR)
                                                    count *= len(cfg.OPTIM.WEIGHT_DECAY)
                                                    count *= len(VALID_BATCH_SIZES)
                                                    count *= len(cfg.OPTIM.MAX_ITER)
                                                    return count
                                                experiment_count = get_experiment_count(cfg)

                                                cur_count = 0
                                                all_hyper_finished = True
                                                all_hyper_dict = {}
                                                # sweep through hyperparameters
                                                for lr in cfg.OPTIM.LR:
                                                    all_hyper_dict[lr] = {}
                                                    for wd in cfg.OPTIM.WEIGHT_DECAY:
                                                        all_hyper_dict[lr][wd] = {}
                                                        for batch_size in VALID_BATCH_SIZES:
                                                            all_hyper_dict[lr][wd][batch_size] = {}
                                                            for iters in cfg.OPTIM.MAX_ITER:
                                                                all_hyper_dict[lr][wd][batch_size][iters] = {}
                                                                cur_count += 1
                                                                hyperparams_str = get_hyperparams_str(
                                                                    cfg.OPTIM.NAME, lr, wd, batch_size, iters)

                                                                # check if experiment has been done
                                                                checkpoint_dir = os.path.join(save_dir, hyperparams_str)
                                                                best_val_path = os.path.join(checkpoint_dir, "best_val.pth")
                                                                last_iter_path = os.path.join(checkpoint_dir, "last_iter.pth")
                                                                if not os.path.exists(best_val_path) or not os.path.exists(last_iter_path):
                                                                    all_dataset_finished = False
                                                                    all_seed_finished = False
                                                                    all_hyper_finished = False
                                                                    import pdb; pdb.set_trace()
                                                                    continue
                                                                else:
                                                                    test_result_dict = {}
                                                                    test_result_path = os.path.join(checkpoint_dir, "test_result.pth")
                                                                    if os.path.exists(test_result_path):
                                                                        print(f"Already exists: {hyperparams_str} {cur_count}/{experiment_count}")
                                                                        test_result_dict = torch.load(test_result_path)
                                                                        for key in ['best_val', 'last_iter']:
                                                                            normhead_keys = [eval_type for eval_type in test_result_dict[key]['test_accs'] if "normhead" in eval_type]
                                                                            for eval_type in normhead_keys:
                                                                                del test_result_dict[key]['test_accs'][eval_type]
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

                                                                            # Create the logreg model and load the weights
                                                                            head, num_classes, in_features = make_classifier_head(
                                                                                cfg.ARCHITECTURE.HEAD, cfg.FEATURE.BACKBONE, cfg.ARCHITECTURE.BIAS, text_dataset)
                                                                            old_logit_head = make_logit_head(
                                                                                head,
                                                                                cfg.LOGIT.FEATURE_NORM,
                                                                                cfg.LOGIT.HEAD_NORM,
                                                                                cfg.LOGIT.USE_LOGIT_SCALE,
                                                                                logit_scale=cfg.LOGIT.LOGIT_SCALE,
                                                                                learn_logit_scale=cfg.LOGIT.LEARN_LOGIT_SCALE,
                                                                                init_learn_logit_scale=cfg.LOGIT.INIT_LEARN_LOGIT_SCALE,
                                                                            )
                                                                            old_logit_head.load_state_dict(result_dict['logit_head'])
                                                                            if old_logit_head.logit_scale is not None:
                                                                                test_result_dict[key]['logit_scale'] = float(old_logit_head.logit_scale.data)
                                                                            else:
                                                                                test_result_dict[key]['logit_scale'] = None

                                                                            zero_shot_weights = get_zero_shot_weights(text_dataset, num_classes, in_features)
                                                                            eval_heads = get_eval_heads(deepcopy(old_logit_head.head), zero_shot_weights)

                                                                            image_encoder = torch.load(image_encoder_path).partial_model
                                                                            image_encoder.load_state_dict(result_dict['image_encoder'])
                                                                            image_encoder = image_encoder.cuda().eval()
                                                                            text_encoder = torch.load(text_encoder_path).partial_model
                                                                            text_encoder.load_state_dict(result_dict['text_encoder'])
                                                                            text_encoder = text_encoder.cuda().eval()

                                                                            for eval_type in eval_heads:
                                                                                eval_head = eval_heads[eval_type]
                                                                                eval_head.cuda().eval()
                                                                                test_loader = DataLoader(
                                                                                    test_dataset,
                                                                                    batch_size=cfg.DATALOADER.TEST_BATCH_SIZE,
                                                                                    shuffle=False,
                                                                                    num_workers=1,
                                                                                    pin_memory=True,
                                                                                )
                                                                                test_acc = validate(eval_head, image_encoder, test_loader, device="cuda")
                                                                                test_result_dict[key]['test_accs'][eval_type] = test_acc
                                                                        torch.save(test_result_dict, test_result_path)
                                                                        print(test_result_dict)
                                                                        print(f"Finished testing {hyperparams_str} {cur_count}/{experiment_count}")
                                                                    all_hyper_dict[lr][wd][batch_size][iters] = test_result_dict   
                                                                
                                                                
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
    import pdb; pdb.set_trace()
    print(f"Saving to {csv_path}")
    save_all_csv(all_headers, all_columns, csv_path)
    print("Done!")

if __name__ == "__main__":
    with torch.no_grad():
        main()