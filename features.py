import os, argparse
import torch

from engine.tools.utils import makedirs, set_random_seed, collect_env_info
from engine.config import get_cfg_default
from engine.transforms.default import build_transform
from engine.datasets.utils import DatasetWrapper, get_few_shot_benchmark
from engine.templates import get_templates
from engine import clip
from engine.clip import partial_model


def get_image_encoder_dir(cfg):
    image_encoder_path = os.path.join(
        cfg.FEATURE_DIR,
        'image',
        "_".join([cfg.FEATURE.BACKBONE, str(cfg.FEATURE.LAYER_IDX)]))
    return image_encoder_path


def get_image_features_path(cfg):
    image_features_path = os.path.join(
        get_image_encoder_dir(cfg), cfg.DATASET.NAME,
        f"shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.pth")
    return image_features_path


def get_test_features_path(cfg):
    test_features_path = os.path.join(
        get_image_encoder_dir(cfg), cfg.DATASET.NAME, "test.pth")
    return test_features_path


def get_text_encoder_dir(cfg):
    text_encoder_path = os.path.join(
        cfg.FEATURE_DIR,
        'text',
        "_".join([cfg.FEATURE.BACKBONE, str(cfg.TEXT_FEATURE.LAYER_IDX)]),)
    return text_encoder_path


def get_text_features_path(cfg):
    text_features_path = os.path.join(
        get_text_encoder_dir(cfg), cfg.DATASET.NAME,
        cfg.TEXT_FEATURE.TEMPLATE + ".pth")
    return text_features_path


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

    # 7. Add configs from input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def extract_text_features(cfg, text_encoder, lab2cname):
    # Extract text features from CLIP
    features_dict = {
        'features': None,
        'labels': None,
        'eot_indices': None,
        'prompts': {},
        'lab2cname': lab2cname,
        'partial_model': None,
    }
    templates = get_templates(cfg.DATASET.NAME, cfg.TEXT_FEATURE.TEMPLATE)
    text_encoder.feature_extractor.eval()
    with torch.no_grad():
        for label, cname in lab2cname.items():
            str_prompts = [template.format(cname.replace("_", " ")) for template in templates]
            prompts = torch.cat([clip.tokenize(p) for p in str_prompts]).cuda()
            features, eot_indices = text_encoder.feature_extractor(prompts)
            features = features.cpu()
            eot_indices = eot_indices.cpu()
            labels = torch.Tensor([label for _ in templates]).long()
            if features_dict['features'] is None:
                features_dict['features'] = features
                features_dict['labels'] = labels
                features_dict['eot_indices'] = eot_indices
            else:
                features_dict['features'] = torch.cat((features_dict['features'], features), 0)
                features_dict['labels'] = torch.cat((features_dict['labels'], labels))
                features_dict['eot_indices'] = torch.cat((features_dict['eot_indices'], eot_indices))
            features_dict['prompts'][label] = str_prompts
    return features_dict



def extract_features(cfg, image_encoder, data_source, transform, num_views=1):
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
    # Start Feature Extractor
    ########################################
    image_encoder.feature_extractor.eval()
    with torch.no_grad():
        for _ in range(num_views):
            for batch_idx, batch in enumerate(loader):
                data = batch["img"].cuda()
                feature = image_encoder.feature_extractor(data) # This is not L2 normed
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

    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    # Check if few-shot indices exist
    few_shot_index_file = os.path.join(
        cfg.FEW_SHOT_DIR, cfg.DATASET.NAME, f"shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.json")
    assert os.path.exists(few_shot_index_file)

    few_shot_benchmark = get_few_shot_benchmark(cfg)

    ########################################
    #   Setup Network
    ########################################
    clip_model, _ = clip.load(cfg.FEATURE.BACKBONE, jit=False)
    clip_model.float()
    clip_model.eval()

    text_encoder_dir = get_text_encoder_dir(cfg)
    makedirs(text_encoder_dir)
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
    # Check if text partial model exists already
    if os.path.exists(text_encoder_path):
        print(f"text encoder already saved at {text_encoder_path}")
        text_encoder = torch.load(text_encoder_path)
    else:
        print(f"Saving text encoder to {text_encoder_path}")
        text_encoder = partial_model.get_text_encoder(cfg, clip_model)
        torch.save(text_encoder, text_encoder_path)

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
            'partial_model': None,
        }
        print(f"Extracting features for texts ...")
        text_features = extract_text_features(
            cfg, text_encoder, few_shot_benchmark['lab2cname'])
        torch.save(text_features, text_features_path)


    image_encoder_dir = get_image_encoder_dir(cfg)
    makedirs(image_encoder_dir)
    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")
    # Check if image partial model exists already
    if os.path.exists(image_encoder_path):
        print(f"Image encoder already saved at {image_encoder_path}")
        image_encoder = torch.load(image_encoder_path)
    else:
        print(f"Saving image encoder to {image_encoder_path}")
        image_encoder = partial_model.get_image_encoder(cfg, clip_model)
        torch.save(image_encoder, image_encoder_path)

    # Check if (image) features are saved already
    image_features_path = get_image_features_path(cfg)

    makedirs(os.path.dirname(image_features_path))
    
    if os.path.exists(image_features_path):
        print(f"Features already saved at {image_features_path}")
    else:
        print(f"Saving features to {image_features_path}")
        image_features = {
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
        image_features['train'] = extract_features(
            cfg, image_encoder, few_shot_benchmark['train'], 
            transform, num_views=cfg.FEATURE.VIEWS_PER_TRAIN)
        
        print(f"Extracting features for val split ...")
        image_features['val'] = extract_features(
            cfg, image_encoder, few_shot_benchmark['val'],
            transform, num_views=cfg.FEATURE.VIEWS_PER_VAL)
    
        torch.save(image_features, image_features_path)

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
            cfg, image_encoder, 
            few_shot_benchmark['test'], test_transform, num_views=1)
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
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)