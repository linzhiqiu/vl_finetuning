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
import random
import os, argparse
import pdb
import torch
torch.set_num_threads(4)
from torch.utils.data import DataLoader
import numpy as np
import csv
from engine.tools.utils import makedirs, set_random_seed, collect_env_info
from engine.config import get_cfg_default
from engine.datasets.utils import TensorDataset, TextTensorDataset, get_label_map
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



# EVAL_MODE = 'wiseft'
EVAL_MODE = 'wiseft_crossmodal'
if EVAL_MODE == 'wiseft':
    #### Partial START
    LOGIT_FOLDER = "logit"
    from logreg_minibatch import get_hyperparams_str, get_save_dir, get_valid_batch_sizes
    EVAL_DIR = "./eval_testset_wiseft"

    IMAGES = [
        "rn50_layer_0",
    ]

    TEXTS = [
        "layer_0",
    ]

    TEMPLATES = [
        "single",
        # "ensemble_all"
    ]

    VIEWS = [
        # "view_10_valview_10_randomcrop",
        "view_1_ccrop",
        # "view_100_valview_100_rcrop",
    ]

    DATASETS_TO_TESTSET = {
        'imagenet': ['imagenetv2', 'imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenet'],
        # 'food101': ['food101_test', 'food101'],
        # 'dtd': ['dtd_test', 'dtd'],
    }

    CROSS_MODALS = [
        # "normtext_ratio_0.5",
        # "normtext_ratio_0.2",
        # "normtext_ratio_0.8",
        # "text_ratio_0.5",
        "text_ratio_0",
    ]

    LOGITS = [
        "linear"
    ]

    HYPERS = [
        # "partial_adamw_fast"
        "adamw_2"
    ]

    ARCHITECTURES = [
        # "linear_zeroshot",
        "linear",
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
elif EVAL_MODE == 'wiseft_crossmodal':
    #### Partial START
    LOGIT_FOLDER = "final_logit"
    from final_logit_ablation import get_hyperparams_str, get_save_dir, get_valid_batch_sizes
    EVAL_DIR = "./eval_testset_wiseft_crossmodal"

    IMAGES = [
        "rn50_layer_0",
    ]

    TEXTS = [
        "layer_0",
    ]

    TEMPLATES = [
        "single",
        # "ensemble_all"
    ]

    VIEWS = [
        # "view_10_valview_10_randomcrop",
        "view_1_ccrop",
        # "view_100_valview_100_rcrop",
    ]

    DATASETS_TO_TESTSET = {
        'imagenet': ['imagenetv2', 'imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenet'],
        # 'food101': ['food101_test', 'food101'],
        # 'dtd': ['dtd_test', 'dtd'],
    }

    CROSS_MODALS = [
        "normtext_ratio_0.5",
        # "normtext_ratio_0.2",
        # "normtext_ratio_0.8",
        # "text_ratio_0.5",
        # "text_ratio_0",
    ]

    LOGITS = [
        "fnorm_True_hnorm_False_logit_Fixed_default",
    ]

    HYPERS = [
        # "partial_adamw_fast"
        "adamw_2"
    ]

    ARCHITECTURES = [
        "linear_zeroshot",
        # "linear",
    ]

    SEEDS = [
        1,
        2,
        # 3,
    ]

    SHOTS = [
        # 1,
        # 2,
        # 4,
        # 8,
        16
    ]


def validate(logit_head, image_encoder, val_loader, device="cuda"):
    logit_head.eval()
    image_encoder.eval()
    val_acc = 0
    val_count = 0.
    for image, image_label in val_loader:
        image = image.to(device)
        image_label = image_label.to(device)
        image_feature = image_encoder(image)
        logit = logit_head(image_feature)
        pred = torch.argmax(logit, dim=1)
        val_acc += torch.sum(pred == image_label).item()
        val_count += image_label.size(0)
    val_acc /= val_count
    return val_acc

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
    cfg.merge_from_file(f"config/{LOGIT_FOLDER}/{logit}.yaml")
    
    # 10. From the hyperparams config file
    cfg.merge_from_file(f"config/hyperparams/logreg_minibatch/{hyper}.yaml")

    # 11. Set the seed
    cfg.SEED = seed

    # # cfg.LOGREG_MINIBATCH_DIR = "/scratch/vl"
    # cfg.LOGREG_MINIBATCH_DIR = "/ssd0/vl"

    cfg.freeze()

    return cfg


def get_wiseft(head, zero_shot_weights, wiseft_ratio=0.5):
    if type(head) == torch.nn.Linear:
        head.weight.data = (1 - wiseft_ratio) * head.weight.data + wiseft_ratio * zero_shot_weights
    elif type(head) == torch.nn.Sequential:
        assert type(head[-1]) == torch.nn.Linear, f"Invalid head: {head}"
        head[-1].weight.data = (1 - wiseft_ratio) * head[-1].weight.data + wiseft_ratio * zero_shot_weights
    return head

def get_eval_heads(head, zero_shot_weights, ratio_list=[0.5]):
    logit_head = make_logit_head(
        deepcopy(head), False, False, False)

    eval_heads = {
        'head': logit_head.cuda().eval(),
    }
    for ratio in ratio_list:
        wiseft = get_wiseft(deepcopy(head), zero_shot_weights, ratio)
        wiseft_head = make_logit_head(
            wiseft, False, False, False)
        eval_heads[f'head_wiseft_{ratio}'] = wiseft_head.cuda().eval()
    return eval_heads

# def get_eval_heads(head):
#     logit_head = make_logit_head(
#         deepcopy(head), False, False, False)


#     eval_heads = {
#         'head': logit_head.cuda().eval(),
#     }
#     return eval_heads

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
    best_val_result = None
    best_test_result = None
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
                    # for key in ['best_val', 'last_iter']:
                    for key in ['best_val']:
                        avg_dict[lr][wd][batchsize][iters][key] = {}
                        std_dict[lr][wd][batchsize][iters][key] = {}

                        avg_dict[lr][wd][batchsize][iters][key]['iter'] = np.mean(
                            [all_seed_dict[seed][lr][wd][batchsize][iters][key]['iter'] for seed in ALL_SEEDS])
                        std_dict[lr][wd][batchsize][iters][key]['iter'] = np.std(
                            [all_seed_dict[seed][lr][wd][batchsize][iters][key]['iter'] for seed in ALL_SEEDS])
                        
                        avg_dict[lr][wd][batchsize][iters][key]['test_accs'] = {}
                        std_dict[lr][wd][batchsize][iters][key]['test_accs'] = {}
                        avg_dict[lr][wd][batchsize][iters][key]['test_accs']['head'] = np.mean(
                            [all_seed_dict[seed][lr][wd][batchsize][iters][key]['test_accs']['head'] for seed in ALL_SEEDS])
                        
                        avg_dict[lr][wd][batchsize][iters][key]['val_acc'] = np.mean(
                            [all_seed_dict[seed][lr][wd][batchsize][iters][key]['val_acc'] for seed in ALL_SEEDS])
                        std_dict[lr][wd][batchsize][iters][key]['val_acc'] = np.std(
                            [all_seed_dict[seed][lr][wd][batchsize][iters][key]['val_acc'] for seed in ALL_SEEDS])
                        std_dict[lr][wd][batchsize][iters][key]['test_accs']['head'] = np.std(
                            [all_seed_dict[seed][lr][wd][batchsize][iters][key]['test_accs']['head'] for seed in ALL_SEEDS])
                        get_best_this_time = False
                        if best_val_result is None or avg_dict[lr][wd][batchsize][iters][key]['val_acc'] > best_val_result:
                            get_best_this_time = True
                        elif avg_dict[lr][wd][batchsize][iters][key]['val_acc'] == best_val_result:
                            print("WARNING: TIE IN VAL ACCURACY")
                            if best_test_result < avg_dict[lr][wd][batchsize][iters][key]['test_accs']['head']:
                                get_best_this_time = True
                        if get_best_this_time:
                            best_test_result = avg_dict[lr][wd][batchsize][iters][key]['test_accs']['head']
                            best_val_result = avg_dict[lr][wd][batchsize][iters][key]['val_acc']
                            best_lr = lr
                            best_wd = wd
                            best_batchsize = batchsize
                            best_iters = iters
                            best_key = key

    columns.append([best_lr, best_wd, best_batchsize, best_iters, best_key,
                    avg_dict[best_lr][best_wd][best_batchsize][best_iters][best_key]['iter'], std_dict[best_lr][best_wd][best_batchsize][best_iters][best_key]['iter'],
                    'head',
                    avg_dict[best_lr][best_wd][best_batchsize][best_iters][best_key]['val_acc'], std_dict[best_lr][best_wd][best_batchsize][best_iters][best_key]['val_acc'],
                    avg_dict[best_lr][best_wd][best_batchsize][best_iters][best_key]['test_accs']['head'], std_dict[best_lr][best_wd][best_batchsize][best_iters][best_key]['test_accs']['head']])
    return header, columns, (best_lr, best_wd, best_batchsize, best_iters, best_key)


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

def take_extra_test_average(all_extra_test_dict):
    ALL_SEEDS = list(all_extra_test_dict.keys())
    ALL_DATASETS = list(all_extra_test_dict[ALL_SEEDS[0]].keys())
    avg_dict = {dataset: [] for dataset in ALL_DATASETS}
    std_dict = {dataset: [] for dataset in ALL_DATASETS}
    for dataset in ALL_DATASETS:
        avg_dict[dataset] = np.mean([all_extra_test_dict[seed][dataset][0] for seed in ALL_SEEDS]), np.mean([all_extra_test_dict[seed][dataset][1] for seed in ALL_SEEDS])
        std_dict[dataset] = np.std([all_extra_test_dict[seed][dataset][0] for seed in ALL_SEEDS]), np.std([all_extra_test_dict[seed][dataset][1] for seed in ALL_SEEDS])
    return avg_dict, std_dict

def save_csv(header,
             columns,
             avg_dict,
             std_dict,
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
    ALL_DATASETS = list(avg_dict.keys())
    all_headers = ['dataset', 'shots', 'image', 'text', 'template', 'view', 'cross_modal', 'architecture', 'logit', 'hyper'] + header + ALL_DATASETS
    all_columns = [[dataset, shots, image, text, template, view, cross_modal, architecture, logit,
                    hyper] + column + [avg_dict[test_dataset] for test_dataset in ALL_DATASETS] for column in columns]
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
                                        for dataset in DATASETS_TO_TESTSET:
                                            print(f"Dataset: {dataset} | {DATASETS_TO_TESTSET[dataset]}")
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
                                                                        try:
                                                                            test_result_dict = torch.load(test_result_path)
                                                                        except:
                                                                            import pdb; pdb.set_trace()
                                                                    else:
                                                                        print(f"Not exists: {hyperparams_str} {cur_count}/{experiment_count}")
                                                                        import pdb; pdb.set_trace()
                                                                    
                                                                    all_hyper_dict[lr][wd][batch_size][iters] = test_result_dict 
                                                                
                                                                
                                                if not all_hyper_finished:
                                                    print(f"Seed {seed} not finished!")
                                                    # break
                                                else:
                                                    all_seed_dict[seed] = all_hyper_dict
                                            
                                            if all_seed_finished:
                                                print(f"Dataset {dataset} finished! Taking average...")
                                                all_dataset_dict[dataset] = take_average(all_seed_dict, cfg.OPTIM.LR, cfg.OPTIM.WEIGHT_DECAY, VALID_BATCH_SIZES, cfg.OPTIM.MAX_ITER)
                                                print(f"Dataset {dataset} finished! Taking average... Done!")
                                                best_lr, best_wd, best_batchsize, best_iters, best_key = all_dataset_dict[dataset][2]
                                                all_extra_test_dict = {}
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

                                                    hyperparams_str = get_hyperparams_str(
                                                        cfg.OPTIM.NAME, best_lr, best_wd, best_batchsize, best_iters)

                                                    save_dir = get_save_dir(cfg)
                                                    checkpoint_dir = os.path.join(save_dir, hyperparams_str)
                                                    best_val_path = os.path.join(checkpoint_dir, "best_val.pth")
                                                    assert os.path.exists(best_val_path)
                                                    result_dict = torch.load(best_val_path)
                                                    # Create the logreg model and load the weights
                                                    head, num_classes, in_features = make_classifier_head(
                                                        cfg.ARCHITECTURE.HEAD, cfg.FEATURE.BACKBONE, cfg.ARCHITECTURE.BIAS, text_dataset)
                                                    old_logit_head = make_logit_head(
                                                        head,
                                                        False,
                                                        False,
                                                        False,
                                                        logit_scale=cfg.LOGIT.LOGIT_SCALE,
                                                        learn_logit_scale=cfg.LOGIT.LEARN_LOGIT_SCALE,
                                                        init_learn_logit_scale=cfg.LOGIT.INIT_LEARN_LOGIT_SCALE,
                                                    )
                                                    old_logit_head.load_state_dict(result_dict['logit_head'])

                                                    zero_shot_weights = get_zero_shot_weights(text_dataset, num_classes, in_features)
                                                    # zero_shot_weights = torch.nn.functional.normalize(zero_shot_weights, dim=1)
                                                    eval_heads = get_eval_heads(deepcopy(old_logit_head.head), zero_shot_weights)
                                                    eval_head = eval_heads['head']
                                                    eval_head_wiseft = eval_heads['head_wiseft_0.5']
                                                    image_encoder_dir = get_image_encoder_dir(cfg)
                                                    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")
                                                    assert os.path.exists(image_encoder_path), image_encoder_path
                                                    image_encoder = torch.load(image_encoder_path).partial_model
                                                    image_encoder.load_state_dict(result_dict['image_encoder'])
                                                    image_encoder = image_encoder.cuda().eval()

                                                    extra_test_path = os.path.join(checkpoint_dir, "extra_test.pth")
                                                    # if os.path.exists(extra_test_path):
                                                    if False:
                                                        extra_test_dict = torch.load(extra_test_path)
                                                    else:
                                                        extra_test_dict = {}
                                                        for test_dataset_name in DATASETS_TO_TESTSET[dataset]:
                                                            test_features_path = os.path.join(
                                                                cfg.FEATURE_DIR,
                                                                'image',
                                                                "_".join([get_backbone_name(cfg), str(cfg.FEATURE.LAYER_IDX)]),
                                                                test_dataset_name,
                                                                "test.pth"
                                                            )
                                                            # import pdb; pdb.set_trace()
                                                            test_features = torch.load(test_features_path)
                                                            test_dataset = TensorDataset(
                                                                test_features['features'], test_features['labels'])
                                                            eval_head.cuda().eval()
                                                            eval_head_wiseft.cuda().eval()
                                                            test_loader = DataLoader(
                                                                test_dataset,
                                                                batch_size=cfg.DATALOADER.TEST_BATCH_SIZE,
                                                                shuffle=False,
                                                                num_workers=1,
                                                                pin_memory=True,
                                                            )
                                                            label_map = get_label_map(cfg, test_dataset_name)
                                                            if label_map is None:
                                                                test_acc = validate(eval_head, image_encoder, test_loader, device="cuda")
                                                                test_acc_wiseft = validate(eval_head_wiseft, image_encoder, test_loader, device="cuda")
                                                            else:
                                                                # change eval_head to use label_map
                                                                assert isinstance(eval_head.head, torch.nn.Linear)
                                                                new_head = deepcopy(eval_head)
                                                                new_linear_head = torch.nn.Linear(eval_head.head.in_features, len(label_map), bias=False).cuda()
                                                                new_linear_head.weight.data = eval_head.head.weight.data[label_map]
                                                                new_head.head = new_linear_head
                                                                test_acc = validate(new_head, image_encoder, test_loader, device="cuda")

                                                                assert isinstance(eval_head_wiseft.head, torch.nn.Linear)
                                                                new_head_wiseft = deepcopy(eval_head_wiseft)
                                                                new_linear_head_wiseft = torch.nn.Linear(eval_head_wiseft.head.in_features, len(label_map), bias=False).cuda()
                                                                new_linear_head_wiseft.weight.data = eval_head_wiseft.head.weight.data[label_map]
                                                                new_head_wiseft.head = new_linear_head_wiseft
                                                                test_acc_wiseft = validate(new_head_wiseft, image_encoder, test_loader, device="cuda")

                                                                
                                                            extra_test_dict[test_dataset_name] = test_acc, test_acc_wiseft

                                                        torch.save(extra_test_dict, extra_test_path)
                                                    all_extra_test_dict[seed] = extra_test_dict
                                                
                                                avg_dict, std_dict = take_extra_test_average(all_extra_test_dict)
                                                    
                                                this_headers, this_columns = save_csv(all_dataset_dict[dataset][0], all_dataset_dict[dataset][1], avg_dict, std_dict,
                                                         os.path.join(result_dir, dataset, "extra_test_results.csv"),
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
    csv_path = os.path.join(EVAL_DIR, f"{dt_string}-{dataset}.csv")
    import pdb; pdb.set_trace()
    print(f"Saving to {csv_path}")
    save_all_csv(all_headers, all_columns, csv_path)
    print("Done!")

if __name__ == "__main__":
    with torch.no_grad():
        main()