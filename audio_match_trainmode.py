import os, argparse
import pdb
import torch
torch.set_num_threads(4)
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
from engine.tools.utils import makedirs
from engine.config import get_cfg_default
from engine.datasets.utils import TensorDataset, TextTensorDataset
from engine.model.head import make_classifier_head
from engine.model.learnable_logit import make_logit_head
from engine import clip
from engine.templates import get_templates
from features import get_few_shot_benchmark, \
                     get_text_encoder_dir, \
                     get_backbone_name, \
                     get_text_encoder_name, \
                     get_image_features_path, \
                     get_test_features_path

EVAL_DIR = "./audio_clip_match_trainmode/"

AUDIO_FEATURES_PATH = "/data3/zhiqiul/vl_finetuning/data/esc-50/ESC-50-master/features_trainmode_batchsize_10.pt"

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


MAP = {
    # 'caltech101' : {
    #     2: "engine",
    #     4: "airplane",
    #     18: "car_horn",
    #     30: "dog",
    #     33: "insects",
    #     0: "laughing",
    #     56: "keyboard_typing",
    #     63: "clock_tick",
    #     70: "chirping_birds",
    #     96: "cat",
    # },
    "imagenet" : {
        8: "hen",
        817: "engine",
        312: "crickets",
        7: "rooster",
        404: "airplane",
        898: "drinking_sipping",
        285: "cat",
        409: "clock_alarm",
        680: "crying_baby",
        673: "mouse_click",
        31: "frog",
        896: "water_drops",
        556: "crackling_fire",
        878: "keyboard_typing",
        308: "insects",
        341: "pig",
        175: "dog",
        691: "breathing",
        882: "vacuum_cleaner",
        473: "can_opening",
        634: "hand_saw",
        892: "clock_tick",
        564: "snoring",
        628: "sea_waves",
        897: "washing_machine",
        725: "pouring_water",
        19: "chirping_birds",
        861: "toilet_flush",
        466: "train",
        879: "rain",
        348: "sheep",
        497: "church_bells",
        491: "chainsaw",
        977: "sea_waves",
        849: "pouring_water",

    }
}
# SUN397_MAP = {
#     "Airplane" : "a/airport/airport"
#     "Hen" : "c/chicken_farm/indoor",
#     "Hen" : "c/chicken_farm/outdoor",
#     "Hen": "c/chicken_coop/indoor",
#     "Hen": "c/chicken_coop/outdoor",
#     "Pig" : "p/pig_farm",
#     "Sea Waves": "c/coast",
#     "Sea Waves": "o/ocean",
#     "Sea Waves": "w/wave",
#     "Keyboard typing": "o/office",
#     "Water Drop": "w/waterfall/cascade",
#     "Water Drop": "w/waterfall/cataract",
#     "Water Drop": "w/waterfall/cascade",
#     "Water Drop": "w/waterfall/block",
#     "Water Drop": "w/waterfall/fan",
#     "Water Drop": "w/waterfall/plunge",
#     "Wind": "w/wind_farm",
#     "Clock tick": "c/clock_tower/indoor",
#     "Train": "r/railroad_track",
#     "Train": "r/railway_yard",
#     "Train": "t/train_railway",
#     "Train": "t/train_station/outdoor",
#     "Train": "t/train_station/platform",
#     "Train": "t/train_station/station",
#     "Toilet Flush": "b/bathroom",
# }


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

IMAGES = [
    "rn50_layer_0",
]

TEXTS = [
    "layer_0",
]

# TEMPLATES = [
#     "single",
# ]

VIEWS = [
    "view_1_ccrop",
]

DATASETS = [
    "caltech101",
    "imagenet",
    # "dtd",
    # "eurosat",
    # "fgvc_aircraft",
    # "food101",
    # "oxford_flowers",
    # "oxford_pets",
    # "stanford_cars",
    "sun397",
    # "ucf101",
]

SEEDS = [
    1,
    2,
    3,
]

# SHOTS = [
#     1,
#     2,
#     4,
#     8,
#     16
# ]


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


def evaluate(cfg, text_datasets, test_dataset):
    results = {}
    for template in text_datasets:
        # Create the model and load the weights
        head, num_classes, in_features = make_classifier_head(
            "linear_zeroshot_norm", cfg.FEATURE.BACKBONE, False, text_datasets)
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
        print(f"Template: {template}, Test Acc: {test_acc}")
    return results


def extract_text_features(cfg, text_encoder, lab2cname, template):
    # Extract text features from CLIP
    features_dict = {
        'features': None,
        'labels': None,
        'eot_indices': None,
        'class_idx_to_text' : None,
        'audioclip_features' : None,
        'sam_features': None,
        'sam_labels': None,
    }

    import sys
    sys.path.append("/data3/zhiqiul/AudioCLIP/")

    from model import AudioCLIP
    aclp = AudioCLIP(
        pretrained=f'/data3/zhiqiul/assets/AudioCLIP-Partial-Training.pt').eval()
    text = [[template.format(cname.replace("_", " "))] for _, cname in lab2cname.items()]
    ((_, _, audioclip_features), _), _ = aclp(text=text)
    features_dict['audioclip_features'] = audioclip_features

    sam_features = torch.load(f'audio_features.pt')
    sam_labels = list(sam_features.keys())
    sam_features = torch.stack(list(sam_features.values()))
    features_dict['sam_audio_features'] = sam_features
    sam_labels = [[l] for l in sam_labels]
    ((_, _, sam_text_features), _), _ = aclp(text=sam_labels)
    features_dict['sam_text_features'] = sam_text_features
    features_dict['sam_labels'] = sam_labels
    
    text_encoder.feature_extractor.eval()
    features_dict['class_idx_to_text'] = {}
    with torch.no_grad():
        for label, cname in lab2cname.items():
            if not label in features_dict['class_idx_to_text']:
                features_dict['class_idx_to_text'][label] = template.format(cname)
            else:
                assert features_dict['class_idx_to_text'][label] == template.format(cname)
            str_prompts = [template.format(cname.replace("_", " "))]
            prompts = torch.cat([clip.tokenize(p) for p in str_prompts]).cuda()
            features, eot_indices = text_encoder.feature_extractor(prompts)
            features = features.cpu()
            eot_indices = eot_indices.cpu()
            labels = torch.Tensor([label]).long()
            if features_dict['features'] is None:
                features_dict['features'] = features
                features_dict['labels'] = labels
                features_dict['eot_indices'] = eot_indices
            else:
                features_dict['features'] = torch.cat((features_dict['features'], features), 0)
                features_dict['labels'] = torch.cat((features_dict['labels'], labels))
                features_dict['eot_indices'] = torch.cat((features_dict['eot_indices'], eot_indices))
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



def get_result_dir(image, template, eval_dir=EVAL_DIR):
    return os.path.join(eval_dir,
                        f"image_{image}_template_{template}")


# def get_result_dir_val(image, 
#                        shots,
#                        view,
#                        eval_dir=EVAL_DIR):
#     return os.path.join(get_result_dir(image, eval_dir),
#                         f"view_{view}_shots_{shots}",)


def main():
    shots = 4
    image = 'rn50_layer_0'
    text = 'layer_0'
    view = "view_1_ccrop"
    template_class = "single"
    # template_class = "classname"
    result_dir = get_result_dir(image, template_class)
    makedirs(result_dir)
    # result_dir_val = get_result_dir_val(image, shots, view)
    # makedirs(result_dir_val)
    all_dataset_dict = {
        # 'val' : {},
        'test' : {},
    }
    for dataset_idx, dataset in enumerate(MAP):
        print(f"Dataset: {dataset} | {dataset_idx + 1}/{len(DATASETS)}")
        cfg = setup_cfg(dataset,
                        shots,
                        image,
                        text,
                        template_class,
                        view,
                        SEEDS[0])
        if torch.cuda.is_available() and cfg.USE_CUDA:
            torch.backends.cudnn.benchmark = True
        
        text_encoder_dir = get_text_encoder_dir(cfg)
        text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
        assert os.path.exists(text_encoder_path), text_encoder_path
        
        text_features_path = get_text_features_path(cfg)
        makedirs(os.path.dirname(text_features_path))
        if os.path.exists(text_features_path):
            print(f"Text features already saved at {text_features_path}")
        else:
            ########################################
            #   Setup Network
            ########################################
            clip_model, _ = clip.load(cfg.FEATURE.BACKBONE, jit=False)
            clip_model.float()
            clip_model.eval()
            few_shot_benchmark = get_few_shot_benchmark(cfg)
            print(f"Saving features to {text_features_path}")
            templates = get_templates(cfg.DATASET.NAME, template_class)
            assert len(templates) == 1, templates
            template = templates[0]
            print(f"Extracting features for texts ...")
            text_encoder = torch.load(text_encoder_path)
            text_features = extract_text_features(
                cfg, text_encoder, few_shot_benchmark['lab2cname'], template)
            torch.save(text_features, text_features_path)

        text_features = torch.load(text_features_path)
        text_features['features'] = torch.nn.functional.normalize(text_features['features'], dim=1)
        text_features['audioclip_features'] = torch.nn.functional.normalize(text_features['audioclip_features'], dim=1)
        text_dataset = TextTensorDataset(
            text_features['audioclip_features'], text_features['labels'], text_features['eot_indices']
        )
        class_idx_to_text = text_features['class_idx_to_text']

        # if True:
        #     print(f"Using Sam's features")
        #     text_features['audioclip_features'] = text_features['sam_text_features']
        #     text_features['labels'] = text_features['sam_labels']
        #     class_idx_to_text = text_features['sam_labels']
        

        audio_features = torch.load(AUDIO_FEATURES_PATH)
        audio_features['features'] = torch.nn.functional.normalize(
            audio_features['features'].reshape(audio_features['features'].shape[0], -1), dim=1)
        audio_filenames = [m['path'] for m in audio_features['meta']]
        class_idx_to_audio = {}
        for m in audio_features['meta']:
            if not m['class'] in class_idx_to_audio:
                class_idx_to_audio[m['class']] = m['classname']
            else:
                assert class_idx_to_audio[m['class']] == m['classname'], (class_idx_to_audio[m['class']], m['classname'])
        audio_dataset = AudioTensorDataset(
            audio_features['features'], audio_features['labels'], audio_filenames
        )

        # logits_audio_text = scale_audio_text * \
        #     audio_features['features'] @ text_features['audioclip_features'].T
        # confidence_audio_text = torch.softmax(logits_audio_text, dim=1)
        # score_audio_text, pred_audio_text = torch.max(confidence_audio_text, dim=1)
        # results = list()
        # for idx, (score, pred, audio_label, audio_file) in enumerate(zip(score_audio_text, pred_audio_text, audio_features['labels'], audio_filenames)):
        #     results.append({
        #         'mode' : 'audio_to_text',
        #         'score' : score.item(),
        #         'pred' : int(pred),
        #         'class' : class_idx_to_text[int(pred)],
        #         'audio' : class_idx_to_audio[audio_label.item()],
        #         'audio_idx' : idx,
        #         'audio_file' : audio_file
        #     })

        # # sort result by score
        # results = sorted(results, key=lambda x: x['score'], reverse=True)
        # # save results as csv file with header
        # df = pd.DataFrame(results)
        # df.to_csv(os.path.join(result_dir, f"{dataset}_audio_text.csv"), index=False, header=True)
        # print(f"Saved results to {os.path.join(result_dir, f'{dataset}_audio_text.csv')}")

        results = list()
        for text_idx in MAP[dataset]:
            audio_label = MAP[dataset][text_idx]
            audio_features_indices = [sample_idx for (sample_idx, audio_idx) in enumerate(audio_features['labels']) 
                                      if class_idx_to_audio[int(audio_idx)] == audio_label]
            selected_audio_features = audio_features['features'][audio_features_indices]
            logits_text_audio = text_features['features'][text_idx].unsqueeze(
                0) @ selected_audio_features.T

            topk_scores, topk_indices = torch.topk(logits_text_audio, k=5, dim=1)
            topk_score, topk_indexs = topk_scores[0], topk_indices[0]
            topk_indexs = topk_indexs.tolist()
            topk_indexs = [audio_features_indices[idx] for idx in topk_indexs]
            result = {
                'mode': 'text_to_topk_audio',
                'class': class_idx_to_text[text_idx],
                'class_idx': text_idx,
            }
            for k_idx, (score, audio_idx) in enumerate(zip(topk_score, topk_indexs)):
                result[f'top{k_idx}_score'] = f"{score.item():.3f}"
                result[f'top{k_idx}_audio'] = class_idx_to_audio[int(audio_features['labels'][int(audio_idx)])]
            for k_idx, (score, audio_idx) in enumerate(zip(topk_score, topk_indexs)):
                result[f'top{k_idx}_file'] = f"{audio_filenames[audio_idx]}"
            results.append(result)
            
        # sort result by score
        results = sorted(results, key=lambda x: x['class_idx'], reverse=False)
        # save results as csv file with header
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(result_dir, f"{dataset}_text_topk_audio.csv"), index=False, header=True)
        print(f"Saved results to {os.path.join(result_dir, f'{dataset}_text_topk_audio.csv')}")

if __name__ == "__main__":
    with torch.no_grad():
        main()