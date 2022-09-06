import os, argparse
import torch

from engine.tools.utils import makedirs, set_random_seed, collect_env_info
from engine.config import get_cfg_default
from engine.transforms.default import build_transform
from engine.datasets.utils import DatasetWrapper, get_few_shot_benchmark
from engine.templates import get_templates
import clip


def get_features_path(cfg):
    features_path = os.path.join(
        cfg.FEATURE_DIR, cfg.DATASET.NAME,
        f"shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}", cfg.FEATURE.NAME+".pth")
    return features_path


def get_test_features_path(cfg):
    test_features_path = os.path.join(
        cfg.FEATURE_DIR, cfg.DATASET.NAME,
        cfg.MODEL.BACKBONE+"_"+str(cfg.FEATURE.LAYER_IDX)+".pth")
    return test_features_path


def get_text_features_path(cfg):
    text_features_path = os.path.join(
        cfg.FEATURE_DIR, cfg.DATASET.NAME,
        cfg.MODEL.BACKBONE+"_"+cfg.FEATURE.TEMPLATE+".pth")
    return text_features_path

def setup_cfg(args):
    cfg = get_cfg_default()

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the few-shot config file
    if args.few_shot_config_file:
        cfg.merge_from_file(args.few_shot_config_file)

    # 3. From the feature-extraction config file
    if args.features_config_file:
        cfg.merge_from_file(args.features_config_file)

    # 4. Add configs from input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def extract_text_features(cfg, lab2cname):
    # Extract text features from CLIP
    features_dict = {
        'features': None,
        'labels': None,
        'prompts': {},
        'lab2cname': lab2cname,
    }
    ########################################
    #   Setup Network
    ########################################
    clip_model, _ = clip.load(cfg.MODEL.BACKBONE, jit=False)
    clip_model.eval()
    
    templates = get_templates(cfg.DATASET.NAME, cfg.FEATURE.TEMPLATE)
    with torch.no_grad():
        for label, cname in lab2cname.items():
            str_prompts = [template.format(cname.replace("_", " ")) for template in templates]
            prompts = torch.cat([clip.tokenize(p) for p in str_prompts]).cuda()
            features = clip_model.encode_text(prompts)
            features = features.cpu()
            labels = torch.Tensor([label for _ in templates]).long()
            if features_dict['features'] is None:
                features_dict['features'] = features
                features_dict['labels'] = labels
            else:
                features_dict['features'] = torch.cat((features_dict['features'], features), 0)
                features_dict['labels'] = torch.cat((features_dict['labels'], labels))
            features_dict['prompts'][label] = str_prompts
        #     text_features = clip_model.encode_text(prompts)
        #     text_features = text_features / \
        #         text_features.norm(dim=-1, keepdim=True)
    return features_dict
    


def extract_features(cfg, data_source, transform, num_views=1):
    if cfg.FEATURE.LAYER_IDX != 0:
        raise NotImplementedError("Only LAYER_IDX=0 is supported for now.")

    features_dict = {
        'features': torch.Tensor(),
        'labels': torch.Tensor(),
        'paths': [],
    }
    ######################################
    #   Setup DataLoader
    ######################################
    loader = torch.utils.data.DataLoader(
        DatasetWrapper(data_source, transform=transform),
        batch_size=cfg.DATALOADER.TEST_BATCH_SIZE,
        sampler=None,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )

    ########################################
    #   Setup Network
    ########################################
    clip_model, _ = clip.load(cfg.MODEL.BACKBONE, jit=False)
    clip_model.eval()

    ########################################
    # Start Feature Extractor
    ########################################
    with torch.no_grad():
        for _ in range(num_views):
            for batch_idx, batch in enumerate(loader):
                data = batch["img"].cuda()
                feature = clip_model.visual(data) # This is not L2 normed
                feature = feature.cpu()
                if batch_idx == 0:
                    features_dict['features'] = feature
                    features_dict['labels'] = batch['label']
                    features_dict['paths'] = batch['impath']
                else:
                    features_dict['features'] = torch.cat((features_dict['features'], feature), 0)
                    features_dict['labels'] = torch.cat((features_dict['labels'], batch['label']))
                    features_dict['paths'] = features_dict['paths'] + list(batch['impath'])
    return features_dict


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    # Check if few-shot indices exist
    few_shot_index_file = os.path.join(
        cfg.FEW_SHOT_DIR, cfg.DATASET.NAME, f"shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.json")
    assert os.path.exists(few_shot_index_file)

    few_shot_benchmark = get_few_shot_benchmark(cfg)

    # Text features extraction
    # (only BACKBONE and DATASET.NAME and FEATURE.TEMPLATE are used)
    text_features_path = get_text_features_path(cfg)

    makedirs(os.path.dirname(text_features_path))

    if os.path.exists(text_features_path):
        print(f"Text features already saved at {text_features_path}")
    else:
        print(f"Saving features to {text_features_path}")
        text_features = {
            'features': torch.Tensor(),
            'labels': torch.Tensor(),
            'prompts': [],
            'classnames': [],
        }
        print(f"Extracting features for texts ...")
        text_features = extract_text_features(
            cfg, few_shot_benchmark['lab2cname'])
        torch.save(text_features, text_features_path)

    # Check if features are saved already
    features_path = get_features_path(cfg)

    makedirs(os.path.dirname(features_path))
    
    if os.path.exists(features_path):
        print(f"Features already saved at {features_path}")
    else:
        print(f"Saving features to {features_path}")
        features = {
            'train': {
                'features': None,
                'labels': None,
                'paths': None,
            },
            'val': {
                'features': None,
                'labels': None,
                'paths': None,
            },
        }
        transform = build_transform(cfg, is_train=True)
        print(f"Extracting features for train split ...")
        features['train'] = extract_features(
            cfg, few_shot_benchmark['train'], 
            transform, num_views=cfg.FEATURE.VIEWS_PER_TRAIN)
        
        print(f"Extracting features for val split ...")
        features['val'] = extract_features(
            cfg, few_shot_benchmark['val'],
            transform, num_views=cfg.FEATURE.VIEWS_PER_VAL)
    
        torch.save(features, features_path)

    # Testset extraction (only BACKBONE and LAYER_IDX matters for testset)
    # Check if features are saved already
    test_features_path = get_test_features_path(cfg)

    makedirs(os.path.dirname(test_features_path))

    if os.path.exists(test_features_path):
        print(f"Test features already saved at {test_features_path}")
    else:
        print(f"Saving features to {test_features_path}")
        test_features = {
            'features': None,
            'labels': None,
            'paths': None,
        }
        test_transform = build_transform(cfg, is_train=False)
        print(f"Extracting features for test split ...")
        test_features = extract_features(
            cfg, few_shot_benchmark['test'], test_transform, num_views=1)
        torch.save(test_features, test_features_path)


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
        "--features-config-file",
        type=str,
        default="",
        help="path to config file for feature extraction",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)