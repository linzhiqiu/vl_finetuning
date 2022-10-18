from copy import deepcopy
import sys
import os, argparse
import pdb
import torch
from eval_single import get_eval_heads
from final_logit_ablation import get_hyperparams_str
torch.set_num_threads(4)
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
from engine.tools.utils import makedirs, set_random_seed
from engine.config import get_cfg_default
from engine.datasets.utils import TensorDataset, TextTensorDataset
from engine.model.head import get_zero_shot_weights, make_classifier_head
from engine.model.learnable_logit import make_logit_head
from engine.optimizer.optim import build_optimizer
from engine.optimizer.scheduler import build_lr_scheduler
from engine import clip
from engine.transforms.default import build_transform
from engine.templates import get_templates
import json
from features import get_few_shot_benchmark, get_image_encoder_dir, \
                     get_text_encoder_dir, \
                     get_text_encoder_name, \
                     get_image_features_path, \
                     get_test_features_path
sys.path.append("/data3/zhiqiul/AudioCLIP/")
from model import AudioCLIP
from utils.transforms import ToTensor1D
import librosa
aclp = AudioCLIP(
    # pretrained=f'/data3/zhiqiul/assets/AudioCLIP-Partial-Training.pt').eval()
    pretrained=f'/data3/zhiqiul/assets/AudioCLIP-Partial-Training.pt').train()
audio_transforms = ToTensor1D()
# derived from ESResNeXt
SAMPLE_RATE = 44100

EVAL_DIR = "./audio_few_shot_results_trainmode/"

ESC_DIR = "/data3/zhiqiul/vl_finetuning/data/esc-50/ESC-50-master/"

scale_audio_image = 54.1855
scale_audio_text = 100.
scale_image_text = 100.

class AudioTensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor, filenames):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.filenames = filenames

    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.filenames[index]

    def __len__(self):
        return self.input_tensor.size(0)


CLASS_MAP = {
    # 'imagenet_26': {
    #     'dataset': 'imagenet',
    #     'class_map': {
    #         8: "1-31251-B-6.wav",  # hen
    #         31: "1-18755-B-4.wav",  # tree frog
    #         94: "2-72547-B-14.wav",  # humming bird
    #         215: "4-184575-A-0.wav",  # Brittany dog
    #         285: "2-82274-B-5.wav",  # Egyptian cat
    #         308: "5-233787-A-7.wav",  # fly
    #         312: "3-129338-A-13.wav",  # cricket
    #         341: "4-250869-A-2.wav",  # pig
    #         348: "2-119161-C-8.wav",  # ram
    #         404: "1-53467-A-47.wav",  # airliner
    #         409: "1-13613-A-37.wav",  # analog clock
    #         466: "2-262579-A-45.wav",  # high-speed tree
    #         473: "5-250753-A-34.wav",  # can opener
    #         491: "3-118657-B-41.wav",  # chainsaw
    #         497: "3-122110-A-46.wav",  # church bells
    #         556: "2-28314-B-12.wav",  # fire screen
    #         634: "3-208820-A-49.wav",  # sawmill -- i used hand saw audio
    #         673: "2-146877-B-31.wav",  # computer mouse
    #         861: "4-152958-A-18.wav",  # toilet seat
    #         878: "3-153444-A-32.wav",  # typewriter keyboard
    #         879: "5-203739-A-10.wav",  # umbrella
    #         882: "5-263902-A-36.wav",  # vacuum cleaner
    #         892: "1-57163-A-38.wav",  # wall clock
    #         896: "1-23094-B-15.wav",  # sink
    #         897: "3-151273-A-35.wav",  # washing machine
    #         898: "5-207836-D-29.wav",  # water bottle -- i used drinking sound
    #     }
    # },
    'imagenet_28': {
        'dataset': 'imagenet',
        'class_map': {
            7: "5-200334-B-1.wav", # rooster
            8: "1-31251-B-6.wav",  # hen
            31: "1-18755-B-4.wav", # tree frog
            19: "2-72547-B-14.wav",  # chickadee
            175: "1-30226-A-0.wav",  # Otterhound
            285: "2-82274-B-5.wav", #  Egyptian cat
            308: "5-233787-A-7.wav", # fly
            312: "3-129338-A-13.wav", # cricket
            341: "4-250869-A-2.wav", # pig
            348: "2-119161-C-8.wav", # ram
            404: "1-53467-A-47.wav", # airliner
            409: "1-13613-A-37.wav", # analog clock
            466: "2-262579-A-45.wav", # high-speed tree
            473: "5-250753-A-34.wav", # can opener
            491: "3-118657-B-41.wav", # chainsaw
            497: "3-122110-A-46.wav", # church bells
            556: "2-28314-B-12.wav", # fire screen
            634: "3-208820-A-49.wav", # sawmill -- i used hand saw audio
            673: "2-146877-B-31.wav", # computer mouse
            861: "4-152958-A-18.wav", # toilet seat
            878: "3-153444-A-32.wav", # typewriter keyboard
            879: "5-203739-A-10.wav", # umbrella
            882: "5-263902-A-36.wav", # vacuum cleaner
            892: "1-57163-A-38.wav", # wall clock
            896: "1-23094-B-15.wav", # sink
            897: "3-151273-A-35.wav", # washing machine
            898: "5-207836-D-29.wav", # water bottle -- i used drinking sound
            977: "5-200461-B-11.wav",# sandbar -- i used sea waves
        }
    },
    # 'imagenet_25': {
    #     'dataset': 'imagenet',
    #     'class_map': {
    #         8: "1-31251-B-6.wav",  # hen
    #         31: "1-18755-B-4.wav",  # tree frog
    #         94: "2-72547-B-14.wav",  # humming bird
    #         285: "2-82274-B-5.wav",  # Egyptian cat
    #         308: "5-233787-A-7.wav",  # fly
    #         312: "3-129338-A-13.wav",  # cricket
    #         341: "4-250869-A-2.wav",  # pig
    #         348: "2-119161-C-8.wav",  # ram
    #         404: "1-53467-A-47.wav",  # airliner
    #         409: "1-13613-A-37.wav",  # analog clock
    #         466: "2-262579-A-45.wav",  # high-speed tree
    #         473: "5-250753-A-34.wav",  # can opener
    #         491: "3-118657-B-41.wav",  # chainsaw
    #         497: "3-122110-A-46.wav",  # church bells
    #         556: "2-28314-B-12.wav",  # fire screen
    #         634: "3-208820-A-49.wav",  # sawmill -- i used hand saw audio
    #         673: "2-146877-B-31.wav",  # computer mouse
    #         861: "4-152958-A-18.wav",  # toilet seat
    #         878: "3-153444-A-32.wav",  # typewriter keyboard
    #         879: "5-203739-A-10.wav",  # umbrella
    #         882: "5-263902-A-36.wav",  # vacuum cleaner
    #         892: "1-57163-A-38.wav",  # wall clock
    #         896: "1-23094-B-15.wav",  # sink
    #         897: "3-151273-A-35.wav",  # washing machine
    #         898: "5-207836-D-29.wav",  # water bottle -- i used drinking sound
    #     }
    # },
    # 'caltech101_10': {
    #     'dataset': 'caltech101',
    #     'class_map': {
    #         96: "4-133047-C-5.wav",  # wild_cat
    #         82: "4-192236-A-0.wav",  # snoopy
    #         50: "3-156581-A-14.wav", # ibis
    #         4: "4-161103-A-47.wav", # airplane
    #         18: "5-180156-B-43.wav", #  car_side
    #         29: "1-79220-A-17.wav", # cup
    #         63: "3-150363-A-38.wav", # metronome
    #         2: "3-141240-A-44.wav", # motorbikes
    #         78: "2-137162-A-11.wav", # schooner
    #         56: "3-153444-A-32.wav", # laptop
    #     }
    # },
    # 'imagenet_16': {
    #     'dataset': 'imagenet',
    #     'class_map': {
    #         8: "5-263831-A-6.wav",  # hen
    #         19: "3-156581-A-14.wav",  # chickadee
    #         31: "3-71964-C-4.wav",  # tree frog
    #         175: "4-192236-A-0.wav",  # Otterhound
    #         285: "3-95694-A-5.wav",  # Egyptian Mau
    #         341: "2-158746-D-2.wav",  # pig
    #         348: "4-188703-D-8.wav",  # ram
    #         743: "1-68670-A-34.wav",  # can opener
    #         497: "2-78381-A-46.wav",  # church
    #         510: "4-172180-A-32.wav",  # keyboard
    #         530: "1-67033-A-37.wav",  # digital clock
    #         673: "2-119139-A-31.wav",  # mouse
    #         779: "5-180156-B-43.wav",  # school bus
    #         849: "2-102414-A-17.wav",  # tea-pot
    #         861: "1-20736-A-18.wav",  # toilet seat
    #         977: "2-137162-A-11.wav",  # sandbar
    #     }
    # },
    # 'imagenet_8': {
    #     'dataset': 'imagenet',
    #     'class_map': {
    #         8: "5-263831-A-6.wav",  # hen
    #         19: "3-156581-A-14.wav",  # chickadee
    #         31: "3-71964-C-4.wav",  # tree frog
    #         175: "4-192236-A-0.wav",  # Otterhound
    #         497: "2-78381-A-46.wav",  # church
    #         530: "1-67033-A-37.wav",  # digital clock
    #         779: "5-180156-B-43.wav",  # school bus
    #         849: "2-102414-A-17.wav",  # tea-pot
    #     }
    # },
    # 'imagenet_27': {
    #     'dataset': 'imagenet',
    #     'class_map': {
    #         7: "5-200334-B-1.wav", # rooster
    #         8: "1-31251-B-6.wav",  # hen
    #         31: "1-18755-B-4.wav", # tree frog
    #         94: "2-72547-B-14.wav", # humming bird
    #         215: "4-184575-A-0.wav", # Brittany dog 
    #         285: "2-82274-B-5.wav", #  Egyptian cat
    #         308: "5-233787-A-7.wav", # fly
    #         312: "3-129338-A-13.wav", # cricket
    #         341: "4-250869-A-2.wav", # pig
    #         348: "2-119161-C-8.wav", # ram
    #         404: "1-53467-A-47.wav", # airliner
    #         409: "1-13613-A-37.wav", # analog clock
    #         466: "2-262579-A-45.wav", # high-speed tree
    #         473: "5-250753-A-34.wav", # can opener
    #         491: "3-118657-B-41.wav", # chainsaw
    #         497: "3-122110-A-46.wav", # church bells
    #         556: "2-28314-B-12.wav", # fire screen
    #         634: "3-208820-A-49.wav", # sawmill -- i used hand saw audio
    #         673: "2-146877-B-31.wav", # computer mouse
    #         861: "4-152958-A-18.wav", # toilet seat
    #         878: "3-153444-A-32.wav", # typewriter keyboard
    #         879: "5-203739-A-10.wav", # umbrella
    #         882: "5-263902-A-36.wav", # vacuum cleaner
    #         892: "1-57163-A-38.wav", # wall clock
    #         896: "1-23094-B-15.wav", # sink
    #         897: "3-151273-A-35.wav", # washing machine
    #         898: "5-207836-D-29.wav", # water bottle -- i used drinking sound
    #     }
    # }
}

def train(logit_head, 
          image_loader, val_loader, audio_loader, test_loader,
          optimizer, scheduler, criterion, iters,
          logit_scale=None,
          eval_freq=100, device="cuda"):
    if image_loader is None and audio_loader is None:
        raise ValueError("Both image_loader and audio_loader are None")
    if image_loader is not None:
        image_loader_iter = iter(image_loader)
    else:
        image_loader_iter = None
    if audio_loader is not None:
        audio_loader_iter = iter(audio_loader)
    else:
        audio_loader_iter = None

    best_val_dict = {
        "iter": None,
        "val_acc": None,
        "image_encoder": None,
        "logit_head": None,
    }

    for i in range(iters):
        logit_head.train()
        if image_loader_iter is not None:
            try:
                image_feature, image_label = next(image_loader_iter)
            except StopIteration:
                image_loader_iter = iter(image_loader)
                image_feature, image_label = next(image_loader_iter)
            image_feature = image_feature.to(device)
            image_label = image_label.to(device)
        else:
            image_feature = None
        
        if audio_loader_iter is not None:
            try:
                audio_feature, audio_label = next(audio_loader_iter)
            except StopIteration:
                audio_loader_iter = iter(audio_loader)
                audio_feature, audio_label = next(audio_loader_iter)
            audio_feature = audio_feature.to(device)
            audio_label = audio_label.to(device)
        else:
            audio_feature = None
        
        if image_feature is not None and audio_feature is not None:
            feature = torch.cat([image_feature, audio_feature], dim=0)
            label = torch.cat([image_label, audio_label], dim=0)
        elif image_feature is not None:
            feature = image_feature
            label = image_label
        elif audio_feature is not None:
            feature = audio_feature
            label = audio_label
        else:
            raise ValueError("Both image_feature and audio_feature are None")

        optimizer.zero_grad()
        logit = logit_head(feature)
        loss = criterion(logit, label)
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if logit_scale is not None:
            if logit_head.logit_scale is not None:
                torch.clamp(logit_head.logit_scale.data, 0, logit_scale)

        if i % eval_freq == 0:
            val_acc = validate(logit_head, val_loader, device=device)
            test_acc = validate(logit_head, test_loader, device=device)
            if best_val_dict["val_acc"] is None or val_acc > best_val_dict["val_acc"]:
                best_val_dict["iter"] = i
                best_val_dict["val_acc"] = val_acc
                best_val_dict['test_acc'] = test_acc
                best_val_dict["logit_head"] = deepcopy(logit_head.state_dict())
    
    val_acc = validate(logit_head, val_loader, device=device)
    test_acc = validate(logit_head, test_loader, device=device)
    last_iter_dict = {
        "iter": i,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "logit_head": deepcopy(logit_head.state_dict()),
    }
    print(f"Best val acc: {best_val_dict['val_acc']:.4f} at iter {best_val_dict['iter']} with test acc {best_val_dict['test_acc']:.4f}")
    print(f"Last iter acc: {last_iter_dict['val_acc']:.4f} at iter {last_iter_dict['iter']} with test acc {last_iter_dict['test_acc']:.4f}")
    return best_val_dict, last_iter_dict



def get_text_features_path(cfg):
    text_features_path = os.path.join(
        EVAL_DIR,
        'text',
        get_text_encoder_name(cfg),
        cfg.DATASET.NAME,
        f"{cfg.TEXT_FEATURE.TEMPLATE}.pth")
    return text_features_path


def save_csv(test_results, test_result_path):
    with open(test_result_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['template', 'accuracy'])
        # Sort the results by accuracy
        for template, acc in sorted(test_results.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([template, acc])


def save_all_csv(all_results, all_result_path):
    results_dict = {
        template: {'mean' : None, 'std' : None, 'accs' : []}
        for template in all_results[0]
    }
    for result in all_results:
        for template in results_dict:
            results_dict['accs'].append(result)
    for template in results_dict:
        results_dict['mean'] = float(np.mean(results_dict['accs']))
        results_dict['std'] = float(np.std(results_dict['accs']))
    with open(all_result_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['template', 'mean', 'std'])
        # Sort the results by accuracy
        for template in sorted(results_dict.keys(), key=lambda x: results_dict[x]['mean'], reverse=True):
            writer.writerow([template, results_dict['mean'], results_dict['std']])


def load_csv(csv_path):
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            results[row[0]] = float(row[1])
    return results


TEMPLATES = [
    "classname",
    "single",
]

SHOTS = 1

SEEDS = [
    1,
    # 2,
    # 3,
]

def validate(logit_head, val_loader, device="cuda"):
    logit_head.eval()
    val_acc = 0
    val_count = 0.
    for image_feature, image_label in val_loader:
        image_feature = image_feature.to(device)
        image_label = image_label.to(device)
        # import pdb; pdb.set_trace()
        logit = logit_head(image_feature)
        pred = torch.argmax(logit, dim=1)
        val_acc += torch.sum(pred == image_label).item()
        val_count += image_label.size(0)
    val_acc /= val_count
    return val_acc


def evaluate(cfg, text_dataset, test_dataset):
    # Create the model and load the weights
    head, num_classes, in_features = make_classifier_head(
        "linear_zeroshot_norm", cfg.FEATURE.BACKBONE, False, text_dataset)
    eval_head = make_logit_head(
        head,
        False,
        False,
        False,
    ).cuda().eval()


    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.DATALOADER.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    test_acc = validate(eval_head, test_loader, device="cuda")
    results = float(test_acc)
    print(f"Test Acc: {test_acc}")
    return results

def transform_features(test_features, sorted_classes):
    # first filter out the classes that are not in sorted_classes
    new_features = None
    new_labels = None
    new_paths = None
    for i in range(test_features['features'].shape[0]):
        if test_features['labels'][i] in sorted_classes:
            if new_features is None:
                new_features = test_features['features'][i].unsqueeze(0)
                new_labels = test_features['labels'][i].unsqueeze(0)
                new_paths = [test_features['paths'][i]]
            else:
                new_features = torch.cat((new_features, test_features['features'][i].unsqueeze(0)), 0)
                new_labels = torch.cat((new_labels, test_features['labels'][i].unsqueeze(0)), 0)
                new_paths += [test_features['paths'][i]]

    # transform new_labels to be the index of sorted_classes
    for i in range(len(new_labels)):
        new_labels[i] = sorted_classes.index(new_labels[i])
    test_features['features'] = new_features
    test_features['labels'] = new_labels
    test_features['paths'] = new_paths
    return test_features


def extract_text_features(cfg, text_encoder, lab2cname, sorted_classes, template):
    # Extract text features from CLIP
    features_dict = {
        'features': None,
        'labels': None,
        'eot_indices': None,
        'class_idx_to_text' : None,
        'audioclip_features' : None,
    }

    sorted_labels = sorted_classes
    text = [[template.format(lab2cname[label].replace("_", " "))] for label in sorted_labels]
    ((_, _, audioclip_features), _), _ = aclp(text=text)
    features_dict['audioclip_features'] = audioclip_features
    assert features_dict['audioclip_features'].shape[0] == len(sorted_labels)
    text_encoder.feature_extractor.eval()
    features_dict['class_idx_to_text'] = {}
    with torch.no_grad():
        for label in sorted_labels:
            cname = lab2cname[label].replace("_", " ")
            if not label in features_dict['class_idx_to_text']:
                features_dict['class_idx_to_text'][label] = template.format(cname)
            else:
                assert features_dict['class_idx_to_text'][label] == template.format(cname)
            str_prompts = [template.format(cname)]
            prompts = torch.cat([clip.tokenize(p) for p in str_prompts]).cuda()
            features, eot_indices = text_encoder.feature_extractor(prompts)
            features = features.cpu()
            eot_indices = eot_indices.cpu()
            labels = torch.Tensor([sorted_labels.index(label)]).long()
            if features_dict['features'] is None:
                features_dict['features'] = features
                features_dict['labels'] = labels
                features_dict['eot_indices'] = eot_indices
            else:
                features_dict['features'] = torch.cat((features_dict['features'], features), 0)
                features_dict['labels'] = torch.cat((features_dict['labels'], labels))
                features_dict['eot_indices'] = torch.cat((features_dict['eot_indices'], eot_indices))
    features_dict['features'] = features_dict['features'] / features_dict['features'].norm(dim=-1, keepdim=True)
    error = (features_dict['audioclip_features'] - features_dict['features']).sum()
    assert error < 0.0001
    return features_dict


def extract_audio_features(sorted_classes, class_map):
    with torch.no_grad():
        # Extract text features from AudioCLIP
        features_dict = {
            'features': None,
            'labels': None,
        }
        track_list = list()
        for label in sorted_classes:
            path_to_audio = f"{ESC_DIR}/audio/{class_map[label]}"
            track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
            if track.shape[0] > 220500:
                track = track[:220500]
            else:
                track = np.pad(track, (0, 220500 - track.shape[0]), 'constant')
            track_list.append(track)
            if features_dict['labels'] is None:
                features_dict['labels'] = torch.Tensor([sorted_classes.index(label)]).long()
            else:
                features_dict['labels'] = torch.cat((features_dict['labels'], torch.Tensor([sorted_classes.index(label)]).long()))
        audio_stacked = torch.stack([audio_transforms(track.reshape(1, -1)) for track in track_list])
        ((audio_features_stacked, _, _), _), _ = aclp(audio=audio_stacked)
        audio_features_stacked = audio_features_stacked / torch.linalg.norm(audio_features_stacked, dim=-1, keepdim=True)
        features_dict['features'] = audio_features_stacked
        # sam_features = torch.load(f'audio_features.pt')
        print("done")
    return features_dict


def setup_cfg(dataset,
              shots,
              image,
              text,
              template,
              view,
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

    # 5. From the template config file
    cfg.merge_from_file(f"config/features/template/{template}.yaml")
    
    # 6. From the augmentation view config file
    cfg.merge_from_file(f"config/features/view/{view}.yaml")
    
    # 6. Set the seed
    cfg.SEED = seed

    cfg.freeze()

    return cfg


def setup_training_cfg(dataset,
                       shots,
                       image,
                       text,
                       template,
                       view,
                       # new
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

    # 5. From the template config file
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

def get_result_dir(view, eval_dir=EVAL_DIR):
    return os.path.join(eval_dir,
                        f"{view}")


# def get_result_dir_val(image, 
#                        shots,
#                        view,
#                        eval_dir=EVAL_DIR):
#     return os.path.join(get_result_dir(image, eval_dir),
#                         f"view_{view}_shots_{shots}",)
def get_text_features_path(text_encoder_dir, dataset, template_class):
    text_features_path = os.path.join(
        text_encoder_dir,
        dataset,
        f"{template_class}.pth")
    return text_features_path


def get_valid_batch_sizes(cfg, othermodal_dataset, image_train_dataset):
    VALID_BATCH_SIZES = []
    for batch_size in cfg.OPTIM.BATCH_SIZE:
        othermodal_batch_size = int(batch_size * cfg.MODALITY.TEXT_BATCH_RATIO)
        image_batch_size = batch_size - othermodal_batch_size
        # check if text batch size is smaller than the size of other modal dataset
        if othermodal_batch_size == 0 or othermodal_batch_size < len(othermodal_dataset):
            # check if image batch size is smaller than the size of image dataset
            if image_batch_size == 0 or image_batch_size < len(image_train_dataset):
                VALID_BATCH_SIZES.append(batch_size)
    if len(VALID_BATCH_SIZES) == 0:
        import pdb; pdb.set_trace()
    print("Valid batch sizes: {}/{}".format(len(VALID_BATCH_SIZES), len(cfg.OPTIM.BATCH_SIZE)))
    return VALID_BATCH_SIZES

def main():
    image = 'rn50_layer_0'
    text = 'layer_0'
    view = "view_1_ccrop"
    cross_modal = 'normtext_ratio_0.5'
    logit = 'fnorm_True_hnorm_False_logit_Fixed_default'
    hyper = 'adamw_small'
    result_dir = get_result_dir(view)
    makedirs(result_dir)
    # result_dir_val = get_result_dir_val(image, shots, view)
    # makedirs(result_dir_val)
    all_dataset_dict = {
        'best_val' : {},
        'last_iter' : {},
        'zero_shot' : {},
        'image_as_classifier': {}
    }
    for dataset_idx, dataset in enumerate(CLASS_MAP):
        original_dataset = CLASS_MAP[dataset]['dataset']
        class_map = CLASS_MAP[dataset]['class_map']
        sorted_classes = sorted(list(class_map.keys()))

        dataset_dir = os.path.join(result_dir, dataset)
        
        # 1: zero-shot with text
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        few_shot_benchmark = None
        for template_class in TEMPLATES:
            cfg = setup_cfg(original_dataset,
                            1, # shots
                            image,
                            text,
                            template_class,
                            view,
                            SEEDS[0])
            text_encoder_dir = get_text_encoder_dir(cfg)
            text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
            assert os.path.exists(text_encoder_path), text_encoder_path
        
            text_features_path = get_text_features_path(text_encoder_dir, dataset, template_class)
            makedirs(os.path.dirname(text_features_path))
            clip_model, _ = clip.load(cfg.FEATURE.BACKBONE, jit=False)
            clip_model.float()
            clip_model.eval()
            if few_shot_benchmark is None:
                few_shot_benchmark = get_few_shot_benchmark(cfg)
            print(f"Saving features to {text_features_path}")
            templates = get_templates(cfg.DATASET.NAME, template_class)
            assert len(templates) == 1, templates
            template = templates[0]
            print(f"Extracting features for texts ...")
            text_encoder = torch.load(text_encoder_path)
            text_features = extract_text_features(
                cfg, text_encoder, few_shot_benchmark['lab2cname'], sorted_classes, template)
            torch.save(text_features, text_features_path)

            text_features = torch.load(text_features_path)
            # should already be normalized, but redo for safety
            text_features['features'] = torch.nn.functional.normalize(text_features['features'], dim=1)
            text_dataset = TextTensorDataset(
                text_features['features'], text_features['labels'], text_features['eot_indices']
            )
            class_idx_to_text = text_features['class_idx_to_text']


            # image testset
            test_features_path = get_test_features_path(cfg)

            makedirs(os.path.dirname(test_features_path))
            assert os.path.exists(test_features_path)
            print(f"Test features already saved at {test_features_path}")
            test_features = torch.load(test_features_path)

            test_features = transform_features(test_features, sorted_classes)

            test_dataset = TensorDataset(
                test_features['features'], test_features['labels'])

            print(f"Template class {template_class} for dataset {dataset} has {len(test_dataset)} test examples")
            test_results = evaluate(cfg, text_dataset, test_dataset)
            all_dataset_dict['zero_shot'][f"{dataset}_template_{template_class}"] = test_results
        # 2: zero-shot with text (single template) # done in for loop above
        # 3: zero-shot with audio
        cfg = setup_cfg(original_dataset,
                        1, # shots
                        image,
                        text,
                        template_class,
                        view,
                        SEEDS[0])

        print(f"Extracting features for audio ...")
        audio_features = extract_audio_features(sorted_classes, class_map)

        # should already be normalized, but redo for safety
        audio_features['features'] = torch.nn.functional.normalize(audio_features['features'], dim=1)
        audio_dataset = TensorDataset(
            audio_features['features'], audio_features['labels']
        )
        # image testset
        test_features_path = get_test_features_path(cfg)
        assert os.path.exists(test_features_path)
        print(f"Test features already saved at {test_features_path}")
        test_features = torch.load(test_features_path)
        test_features = transform_features(test_features, sorted_classes)
        test_dataset = TensorDataset(
            test_features['features'], test_features['labels'])
        print(f"Audio zero-shot for dataset {dataset} has {len(test_dataset)} test examples")
        test_results = evaluate(cfg, audio_dataset, test_dataset)
        all_dataset_dict['zero_shot'][f"{dataset}_audio"] = test_results

        crossmodal_features = torch.cat((audio_features['features'], text_features['features']), 0)
        crossmodal_labels = torch.cat((audio_features['labels'], text_features['labels']), 0)
        crossmodal_dataset = TensorDataset(
            crossmodal_features, crossmodal_labels
        )

        # 4: few-shot with images only (1-shot)
        for shots in [1, 2, 4]:
            for architecture, cross_modal in [('linear_zeroshot', 'normtext_ratio_0.5')]:
                if True:
                    cfg = setup_training_cfg(original_dataset,
                                             shots,  # shots
                                            image,
                                            text,
                                            template_class,
                                            view,
                                            cross_modal,
                                            architecture,
                                            logit,
                                            hyper,
                                            SEEDS[0])
                    if cfg.SEED >= 0:
                        print("Setting fixed seed: {}".format(cfg.SEED))
                        set_random_seed(cfg.SEED)
                    image_encoder_dir = get_image_encoder_dir(cfg)
                    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")

                    image_features_path = get_image_features_path(cfg)
                    image_features = torch.load(image_features_path)
                    image_features['train'] = transform_features(image_features['train'], sorted_classes)
                    image_features['val'] = transform_features(image_features['val'], sorted_classes)
                    image_train_dataset = TensorDataset(
                        image_features['train']['features'], image_features['train']['labels'])
                    image_val_dataset = TensorDataset(
                        image_features['val']['features'], image_features['val']['labels'])
                    
                    save_dir = os.path.join(dataset_dir, f'{shots}_shot_{architecture}_{cross_modal}_audioandtext')
                    makedirs(save_dir)
                    # filter out invalid batch sizes
                    VALID_BATCH_SIZES = get_valid_batch_sizes(
                        cfg, crossmodal_dataset, image_train_dataset)

                    def get_experiment_count(cfg):
                        count = 1
                        count *= len(cfg.OPTIM.LR)
                        count *= len(cfg.OPTIM.WEIGHT_DECAY)
                        count *= len(VALID_BATCH_SIZES)
                        count *= len(cfg.OPTIM.MAX_ITER)
                        return count
                    experiment_count = get_experiment_count(cfg)
                    cur_count = 0
                    best_val = None
                    best_test = None
                    best_val_last_iter = None
                    best_test_last_iter = None
                    # sweep through hyperparameters
                    for lr in cfg.OPTIM.LR:
                        for wd in cfg.OPTIM.WEIGHT_DECAY:
                            for batch_size in VALID_BATCH_SIZES:
                                for iters in cfg.OPTIM.MAX_ITER:
                                    cur_count += 1

                                    hyperparams_str = get_hyperparams_str(
                                        cfg.OPTIM.NAME, lr, wd, batch_size, iters)
                                    
                                    # check if experiment has been done
                                    checkpoint_dir = os.path.join(save_dir, hyperparams_str)
                                    makedirs(checkpoint_dir)
                                    best_val_path = os.path.join(checkpoint_dir, "best_val.pth")
                                    last_iter_path = os.path.join(checkpoint_dir, "last_iter.pth")
                                    # if os.path.exists(best_val_path) and os.path.exists(last_iter_path):
                                    #     print(f"Hyperparameters [{cur_count}/{experiment_count}]: {hyperparams_str}. Already Done")
                                    #     continue
                                    # else:
                                    print(f"{cross_modal} {architecture} [{cur_count}/{experiment_count}]: {hyperparams_str}. Running")

                                    # Create the logreg model
                                    head, num_classes, in_features = make_classifier_head(
                                        cfg.ARCHITECTURE.HEAD, cfg.FEATURE.BACKBONE, cfg.ARCHITECTURE.BIAS, text_dataset) # NOTE: always use text to zero-shot init
                                    logit_head = make_logit_head(
                                        head,
                                        cfg.LOGIT.FEATURE_NORM,
                                        cfg.LOGIT.HEAD_NORM,
                                        cfg.LOGIT.USE_LOGIT_SCALE,
                                        logit_scale=cfg.LOGIT.LOGIT_SCALE,
                                        learn_logit_scale=cfg.LOGIT.LEARN_LOGIT_SCALE,
                                        init_learn_logit_scale=cfg.LOGIT.INIT_LEARN_LOGIT_SCALE,
                                    ).train().cuda()
                                    # Create the optimizer
                                    params_groups = [
                                        {'params': logit_head.parameters()},
                                    ]
                                    optimizer = build_optimizer(params_groups, cfg, cfg.OPTIM.NAME, lr, wd)
                                    scheduler = build_lr_scheduler(
                                        optimizer,
                                        cfg.OPTIM.LR_SCHEDULER,
                                        cfg.OPTIM.WARMUP_ITER,
                                        iters,
                                        warmup_type=cfg.OPTIM.WARMUP_TYPE,
                                        warmup_lr=cfg.OPTIM.WARMUP_MIN_LR
                                    )
                                    criterion = torch.nn.CrossEntropyLoss()

                                    crossmodal_batch_size = int(batch_size * cfg.MODALITY.TEXT_BATCH_RATIO)
                                    image_batch_size = batch_size - crossmodal_batch_size

                                    crossmodal_loader = None
                                    if crossmodal_batch_size > 0:
                                        crossmodal_loader = DataLoader(
                                            crossmodal_dataset,
                                            batch_size=crossmodal_batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=True,
                                            drop_last=True,
                                        )
                                    
                                    image_loader = None
                                    if image_batch_size > 0:
                                        image_loader = DataLoader(
                                            image_train_dataset,
                                            batch_size=image_batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=True,
                                            drop_last=True,
                                        )
                                    
                                    val_loader = DataLoader(
                                        image_val_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True,
                                    )

                                    test_loader = DataLoader(
                                        test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True,
                                    )

                                    best_val_dict, last_iter_dict = train(
                                        logit_head,
                                        image_loader, val_loader, crossmodal_loader, test_loader,
                                        optimizer, scheduler, criterion, iters,
                                        logit_scale=cfg.LOGIT.LOGIT_SCALE,
                                        eval_freq=cfg.OPTIM.EVAL_FREQ)
                                    
                                    if best_val is None or best_val_dict['val_acc'] > best_val:
                                        best_val = best_val_dict['val_acc']
                                        best_test = best_val_dict['test_acc']
                                    elif best_val_dict['val_acc'] == best_val:
                                        best_test = max(best_test, best_val_dict['test_acc'])
                                    
                                    if best_val_last_iter is None or last_iter_dict['val_acc'] > best_val_last_iter:
                                        best_val_last_iter = last_iter_dict['val_acc']
                                        best_test_last_iter = last_iter_dict['test_acc']
                                    elif last_iter_dict['val_acc'] == best_val_last_iter:
                                        best_test_last_iter = max(best_test_last_iter, last_iter_dict['test_acc'])
                                    
                                    torch.save(best_val_dict, best_val_path)
                                    torch.save(last_iter_dict, last_iter_path)
                    print(f"For {cross_modal} {architecture} {dataset} {shots}")
                    print(f"Best Val: {best_val} Best Test: {best_test}")
                    print(f"Best Val Last Iter: {best_val_last_iter} Best Test Last Iter: {best_test_last_iter}")
                    all_dataset_dict['best_val'][f"{dataset}_{architecture}_{cross_modal}_shot_{shots}"] = {'val_acc': best_val, 'test_acc': best_test}
                    all_dataset_dict['last_iter'][f"{dataset}_{architecture}_{cross_modal}_shot_{shots}"] = {
                        'val_acc': best_val_last_iter, 'test_acc': best_test_last_iter}
    all_dataset_path = os.path.join(result_dir, 'all_dataset_dict_with_text.pt')
    print(json.dumps(all_dataset_dict, indent=2))
    import pdb; pdb.set_trace()
    torch.save(all_dataset_dict, all_dataset_path)

if __name__ == "__main__":
    main()