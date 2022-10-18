import os, argparse
import torch

from engine.tools.utils import makedirs, set_random_seed, collect_env_info
from engine.config import get_cfg_default
from engine.transforms.default import build_transform
from engine.datasets.utils import DatasetWrapper, get_testset
from engine.templates import get_templates
from engine import clip
from engine.clip import partial_model


def get_backbone_name(cfg):
    return cfg.FEATURE.BACKBONE.replace("/", "-")


def get_image_encoder_name(cfg):
    return "_".join([get_backbone_name(cfg), str(cfg.FEATURE.LAYER_IDX)])


def get_image_encoder_dir(cfg):
    image_encoder_path = os.path.join(
        cfg.FEATURE_DIR,
        'image',
        get_image_encoder_name(cfg)
    )
    return image_encoder_path


def get_test_features_path(cfg):
    test_features_path = os.path.join(
        get_image_encoder_dir(cfg),
        cfg.DATASET.NAME,
        "test.pth"
    )
    return test_features_path


def setup_cfg(args):
    cfg = get_cfg_default()

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the image encoder config file
    if args.image_encoder_config_file:
        cfg.merge_from_file(args.image_encoder_config_file)

    # 3. From the augmentation view config file
    if args.view_config_file:
        cfg.merge_from_file(args.view_config_file)

    # 4. Add configs from input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def extract_features(cfg, image_encoder, data_source, num_views=1):
    features_dict = {
        'features': torch.Tensor(),
        'labels': torch.Tensor(),
    }
    ######################################
    #   Setup DataLoader
    ######################################
    loader = torch.utils.data.DataLoader(
        data_source,
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
                data, label = batch
                data = data.cuda()
                feature = image_encoder.feature_extractor(data) # This is not L2 normed
                feature = feature.cpu()
                if batch_idx == 0:
                    features_dict['features'] = feature
                    features_dict['labels'] = label
                else:
                    features_dict['features'] = torch.cat((features_dict['features'], feature), 0)
                    features_dict['labels'] = torch.cat((features_dict['labels'], label))
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

    ########################################
    #   Setup Network
    ########################################
    clip_model, _ = clip.load(cfg.FEATURE.BACKBONE, jit=False)
    clip_model.float()
    clip_model.eval()

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


    # Testset extraction (only BACKBONE and LAYER_IDX matters for testset)
    # Check if features are saved already
    test_features_path = get_test_features_path(cfg)

    makedirs(os.path.dirname(test_features_path))
    # if os.path.exists(test_features_path):
    if False:
        print(f"Test features already saved at {test_features_path}")
    else:
        test_transform = build_transform(cfg, is_train=False)
        testset = get_testset(cfg, test_transform)
        print(f"Saving features to {test_features_path}")
        test_features = {
            'features': None,
            'labels': None,
            'paths': None,
        }
        print(f"Extracting features for test split ...")
        test_features = extract_features(
            cfg, image_encoder, 
            testset, num_views=1)
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
        "--image-encoder-config-file",
        type=str,
        default="",
        help="path to config file for image feature encoder",
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