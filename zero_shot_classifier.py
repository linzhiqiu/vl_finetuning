import os, argparse
import pdb
import torch
torch.set_num_threads(4)
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import csv
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

EVAL_DIR = "./zero_shot_clip/"


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
            results_dict[template]['accs'].append(result[template])
    for template in results_dict:
        results_dict[template]['mean'] = float(np.mean(results_dict[template]['accs']))
        results_dict[template]['std'] = float(np.std(results_dict[template]['accs']))
    with open(all_result_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['template', 'mean', 'std'])
        # Sort the results by accuracy
        for template in sorted(results_dict.keys(), key=lambda x: results_dict[x]['mean'], reverse=True):
            writer.writerow([template, results_dict[template]['mean'], results_dict[template]['std']])


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
            "linear_zeroshot_norm", cfg.FEATURE.BACKBONE, False, text_datasets[template])
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
        results[template] = float(test_acc)
        print(f"Template: {template}, Test Acc: {test_acc}")
    return results


def extract_text_features(cfg, text_encoder, lab2cname, templates):
    # Extract text features from CLIP
    features_dict = {
        template: {
            'features': None,
            'labels': None,
            'eot_indices': None,
        } for template in templates
    }
    text_encoder.feature_extractor.eval()
    with torch.no_grad():
        for template in tqdm(templates):
            for label, cname in lab2cname.items():
                str_prompts = [template.format(cname.replace("_", " "))]
                prompts = torch.cat([clip.tokenize(p) for p in str_prompts]).cuda()
                features, eot_indices = text_encoder.feature_extractor(prompts)
                features = features.cpu()
                eot_indices = eot_indices.cpu()
                labels = torch.Tensor([label]).long()
                if features_dict[template]['features'] is None:
                    features_dict[template]['features'] = features
                    features_dict[template]['labels'] = labels
                    features_dict[template]['eot_indices'] = eot_indices
                else:
                    features_dict[template]['features'] = torch.cat((features_dict[template]['features'], features), 0)
                    features_dict[template]['labels'] = torch.cat((features_dict[template]['labels'], labels))
                    features_dict[template]['eot_indices'] = torch.cat((features_dict[template]['eot_indices'], eot_indices))
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



def get_result_dir(image, eval_dir=EVAL_DIR):
    return os.path.join(eval_dir,
                        f"image_{image}")


def get_result_dir_val(image, 
                       shots,
                       view,
                       eval_dir=EVAL_DIR):
    return os.path.join(get_result_dir(image, eval_dir),
                        f"view_{view}_shots_{shots}",)


def main():
    shots = 4
    image = 'rn50_layer_0'
    text = 'layer_0'
    view = "view_1_ccrop"
    template = "ensemble"
    result_dir = get_result_dir(image)
    result_dir_val = get_result_dir_val(image, shots, view)
    makedirs(result_dir)
    makedirs(result_dir_val)
    all_dataset_dict = {
        'val' : {},
        'test' : {},
    }
    for dataset_idx, dataset in enumerate(DATASETS):
        print(f"Dataset: {dataset} | {dataset_idx + 1}/{len(DATASETS)}")
        val_results_all_seeds = []
        for seed in SEEDS:
            print(f"Seed: {seed}")
            cfg = setup_cfg(dataset,
                            shots,
                            image,
                            text,
                            template,
                            view,
                            seed)
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
                templates = get_templates(cfg.DATASET.NAME, template)
                print(f"Extracting features for texts ...")
                text_encoder = torch.load(text_encoder_path)
                text_features = extract_text_features(
                    cfg, text_encoder, few_shot_benchmark['lab2cname'], templates)
                torch.save(text_features, text_features_path)

            text_features = torch.load(text_features_path)
            text_datasets = {template : TextTensorDataset(
                text_features[template]['features'], text_features[template]['labels'], text_features[template]['eot_indices'])
                for template in text_features
            }

            # eval on test set
            test_result_path = os.path.join(result_dir, dataset, 'test_result.csv')
            makedirs(os.path.dirname(test_result_path))
            if os.path.exists(test_result_path):
                print(f"Test result already saved at {test_result_path}")
                test_results = load_csv(test_result_path)
            else:
                print(f"Saving test result to {test_result_path}")
                test_features_path = get_test_features_path(cfg)
                test_features = torch.load(test_features_path)
                test_dataset = TensorDataset(
                    test_features['features'], test_features['labels'])

                test_results = evaluate(cfg, text_datasets, test_dataset)
                save_csv(test_results, test_result_path)
            all_dataset_dict['test'][dataset] = test_results
            # eval on val set
            val_result_path = os.path.join(result_dir_val, dataset, f"seed_{seed}", 'val_result.csv')
            makedirs(os.path.dirname(val_result_path))
            if os.path.exists(val_result_path):
                print(f"Val result already saved at {val_result_path}")
                val_results = load_csv(val_result_path)
            else:
                print(f"Saving val result to {val_result_path}")
                image_features_path = get_image_features_path(cfg)
                image_features = torch.load(image_features_path)
                val_dataset = TensorDataset(
                    image_features['val']['features'], image_features['val']['labels'])

                val_results = evaluate(cfg, text_datasets, val_dataset)
                save_csv(val_results, val_result_path)
            val_results_all_seeds.append(val_results)
        val_all_result_path = os.path.join(result_dir_val, dataset, 'val_result.csv')
        if os.path.exists(val_all_result_path):
            print(f"Val result already saved at {val_all_result_path}")
        else:
            print(f"Saving val result to {val_all_result_path}")
            save_all_csv(val_results_all_seeds, val_all_result_path)
        val_all_result = load_csv(val_all_result_path)
        all_dataset_dict['val'][dataset] = val_all_result
    
    all_dataset_val_result_path = os.path.join(result_dir, 'all_dataset_val_result.csv')
    # average accuracy over all datasets
    all_dataset_val_result = {}
    for dataset in all_dataset_dict['val']:
        for key in all_dataset_dict['val'][dataset]:
            if key not in all_dataset_val_result:
                all_dataset_val_result[key] = []
            all_dataset_val_result[key].append(all_dataset_dict['val'][dataset][key])
    for key in all_dataset_val_result:
        all_dataset_val_result[key] = float(np.mean(all_dataset_val_result[key]))
    save_csv(all_dataset_val_result, all_dataset_val_result_path)
    all_dataset_test_result_path = os.path.join(result_dir, 'all_dataset_test_result.csv')
    # average accuracy over all datasets
    all_dataset_test_result = {}
    for dataset in all_dataset_dict['test']:
        for key in all_dataset_dict['test'][dataset]:
            if key not in all_dataset_test_result:
                all_dataset_test_result[key] = []
            all_dataset_test_result[key].append(all_dataset_dict['test'][dataset][key])
    for key in all_dataset_test_result:
        all_dataset_test_result[key] = float(np.mean(all_dataset_test_result[key]))
    save_csv(all_dataset_test_result, all_dataset_test_result_path)

if __name__ == "__main__":
    with torch.no_grad():
        main()