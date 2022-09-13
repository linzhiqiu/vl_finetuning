from copy import deepcopy
import os, argparse
import torch
from torch.utils.data import DataLoader

from engine.tools.utils import makedirs, set_random_seed, collect_env_info
from engine.config import get_cfg_default
from engine.datasets.utils import TensorDataset, TextTensorDataset
from engine.model.head import make_classifier_head
from engine.model.logit import make_logit_head
from engine.optimizer.optim import build_optimizer
from engine.optimizer.scheduler import build_lr_scheduler
from features import get_backbone_name, \
                     get_few_shot_setup_name, \
                     get_view_name, \
                     get_image_features_path, \
                     get_text_features_path, \
                     get_test_features_path, \
                     get_image_encoder_dir, \
                     get_text_encoder_dir
                     

def get_benchmark_name(cfg):
    benchmark_name = "-".join([cfg.DATASET.NAME, get_few_shot_setup_name(cfg)])
    return benchmark_name


def get_text_feature_name(cfg):
    text_feature_name = "-".join([cfg.TEXT_FEATURE.TEMPLATE, f"txlayer_{cfg.TEXT_FEATURE.LAYER_IDX}"])
    return text_feature_name


def get_image_feature_name(cfg):
    image_feature_name = "-".join([get_view_name(cfg), f"imlayer_{cfg.FEATURE.LAYER_IDX}"])
    return image_feature_name


def get_cross_modal_name(cfg):
    text_feature_name = get_text_feature_name(cfg)
    image_feature_name = get_image_feature_name(cfg)
    if cfg.MODALITY.TEXT_BATCH_RATIO == 0:
        feature_name = image_feature_name
    elif cfg.MODALITY.TEXT_BATCH_RATIO == 1:
        feature_name = text_feature_name
    else:
        feature_name = f"{text_feature_name}-{image_feature_name}-textratio_{cfg.MODALITY.TEXT_BATCH_RATIO}"
    return os.path.join(
        get_backbone_name(cfg),
        feature_name
    )


def get_architecture_name(cfg):
    bias_str = "_bias" if cfg.ARCHITECTURE.BIAS else ""
    return cfg.ARCHITECTURE.HEAD + bias_str


def get_logit_name(cfg):
    name = "logit_"
    if cfg.LOGIT.FEATURE_NORM and cfg.LOGIT.HEAD_NORM:
        name += "cosine"
        if cfg.LOGIT.USE_LOGIT_SCALE:
            name += "_logit_scale"
    elif not cfg.LOGIT.FEATURE_NORM and not cfg.LOGIT.HEAD_NORM:
        name += "linear"
    else:
        raise NotImplementedError()
    return name


def get_save_dir(cfg):
    save_dir = os.path.join(
        cfg.LOGREG_MINIBATCH_DIR,
        get_benchmark_name(cfg),
        get_cross_modal_name(cfg),
        get_architecture_name(cfg),
        get_logit_name(cfg),
    )
    return save_dir


def get_hyperparams_str(optim,
                        lr,
                        wd,
                        batch_size,
                        iters):
    hyperparams_str = f"optim_{optim}-lr_{lr}-wd_{wd}-bs_{batch_size}-iters_{iters}"
    return hyperparams_str


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
    
    # 7. From the cross-modal config file
    if args.cross_modal_config_file:
        cfg.merge_from_file(args.cross_modal_config_file)
    
    # 8. From the architecture config file
    if args.architecture_config_file:
        cfg.merge_from_file(args.architecture_config_file)

    # 9. From the logit config file
    if args.logit_config_file:
        cfg.merge_from_file(args.logit_config_file)
    
    # 10. From the hyperparams config file
    if args.hyperparams_config_file:
        cfg.merge_from_file(args.hyperparams_config_file)

    # 11. Add configs from input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def train(logit_head, image_encoder, text_encoder,
          image_loader, val_loader, text_loader,
          optimizer, scheduler, criterion, iters,
          eval_freq=100, device="cuda"):
    if image_loader is None and text_loader is None:
        raise ValueError("Both image_loader and text_loader are None")
    if image_loader is not None:
        image_loader_iter = iter(image_loader)
    else:
        image_loader_iter = None
    if text_loader is not None:
        text_loader_iter = iter(text_loader)
    else:
        text_loader_iter = None

    best_val_dict = {
        "iter": None,
        "val_acc": None,
        "image_encoder": None,
        "text_encoder": None,
        "logit_head": None,
    }

    for i in range(iters):
        logit_head.train()
        image_encoder.train()
        text_encoder.train()
        if image_loader_iter is not None:
            try:
                image, image_label = next(image_loader_iter)
            except StopIteration:
                image_loader_iter = iter(image_loader)
                image, image_label = next(image_loader_iter)
            image = image.to(device)
            image_label = image_label.to(device)
            image_feature = image_encoder(image)
        else:
            image_feature = None
        
        if text_loader_iter is not None:
            try:
                text, text_label, eot_indices = next(text_loader_iter)
            except StopIteration:
                text_loader_iter = iter(text_loader)
                text, text_label, eot_indices = next(text_loader_iter)
            text = text.to(device)
            text_label = text_label.to(device)
            eot_indices = eot_indices.to(device)
            text_feature = text_encoder(text, eot_indices)
        else:
            text_feature = None
        
        if image_feature is not None and text_feature is not None:
            feature = torch.cat([image_feature, text_feature], dim=0)
            label = torch.cat([image_label, text_label], dim=0)
        elif image_feature is not None:
            feature = image_feature
            label = image_label
        elif text_feature is not None:
            feature = text_feature
            label = text_label
        else:
            raise ValueError("Both image_feature and text_feature are None")

        logit = logit_head(feature)
        loss = criterion(logit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % eval_freq == 0:
            val_acc = validate(logit_head, image_encoder, val_loader, device=device)
            if best_val_dict["val_acc"] is None or val_acc > best_val_dict["val_acc"]:
                best_val_dict["iter"] = i
                best_val_dict["val_acc"] = val_acc
                best_val_dict["image_encoder"] = deepcopy(image_encoder.state_dict())
                best_val_dict["text_encoder"] = deepcopy(text_encoder.state_dict())
                best_val_dict["logit_head"] = deepcopy(logit_head.state_dict())
    
    val_acc = validate(logit_head, image_encoder, val_loader, device=device)
    last_iter_dict = {
        "iter": i,
        "val_acc": val_acc,
        "image_encoder": deepcopy(image_encoder.state_dict()),
        "text_encoder": deepcopy(text_encoder.state_dict()),
        "logit_head": deepcopy(logit_head.state_dict()),
    }
    return best_val_dict, last_iter_dict
            

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


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))
    image_encoder_dir = get_image_encoder_dir(cfg)
    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")

    text_encoder_dir = get_text_encoder_dir(cfg)
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")

    text_features_path = get_text_features_path(cfg)
    text_features = torch.load(text_features_path)
    text_dataset = TextTensorDataset(
        text_features['features'], text_features['labels'], text_features['eot_indices'])

    image_features_path = get_image_features_path(cfg)
    image_features = torch.load(image_features_path)
    image_train_dataset = TensorDataset(
        image_features['train']['features'], image_features['train']['labels'])
    image_val_dataset = TensorDataset(
        image_features['val']['features'], image_features['val']['labels'])

    # test_features_path = get_test_features_path(cfg)
    # test_features = torch.load(test_features_path)
    # test_dataset = TensorDataset(
    #     test_features['features'], test_features['labels'])
    
    save_dir = get_save_dir(cfg)

    # filter out invalid batch sizes
    _, num_classes, _ = make_classifier_head(
        cfg, text_dataset)
    VALID_BATCH_SIZES = []
    for batch_size in cfg.OPTIM.BATCH_SIZE:
        text_batch_size = int(batch_size * cfg.MODALITY.TEXT_BATCH_RATIO)
        image_batch_size = batch_size - text_batch_size
        # check if text batch size is smaller than the size of text dataset
        if text_batch_size == 0 or text_batch_size < len(text_dataset):
            # check if image batch size is smaller than the size of image dataset
            if image_batch_size == 0 or image_batch_size < len(image_train_dataset):
                VALID_BATCH_SIZES.append(batch_size)
    if len(VALID_BATCH_SIZES) == 0:
        import pdb; pdb.set_trace()
    print("Valid batch sizes: {}/{}".format(len(VALID_BATCH_SIZES), len(cfg.OPTIM.BATCH_SIZE)))

    def get_experiment_count(cfg):
        count = 1
        count *= len(cfg.OPTIM.LR)
        count *= len(cfg.OPTIM.WEIGHT_DECAY)
        count *= len(VALID_BATCH_SIZES)
        count *= len(cfg.OPTIM.MAX_ITER)
        return count
    experiment_count = get_experiment_count(cfg)
    cur_count = 0
    # sweep through hyperparameters
    for lr in cfg.OPTIM.LR:
        for wd in cfg.OPTIM.WEIGHT_DECAY:
            for batch_size in VALID_BATCH_SIZES:
                for iters in cfg.OPTIM.MAX_ITER:
                    cur_count += 1

                    hyperparams_str = get_hyperparams_str(
                        cfg.OPTIM.NAME, lr, wd, batch_size, iters)
                    print(f"Hyperparameters [{cur_count}/{experiment_count}]: {hyperparams_str}")
                    
                    # train logreg

                    # Create the logreg model
                    head, num_classes, in_features = make_classifier_head(
                        cfg, text_dataset)
                    logit_head = make_logit_head(
                        cfg, head
                    ).train().cuda()

                    image_encoder = torch.load(
                        image_encoder_path).partial_model.train().cuda()
                    text_encoder = torch.load(
                        text_encoder_path).partial_model.train().cuda()
                    # Create the optimizer
                    params_groups = [
                        {'params': logit_head.parameters()},
                        {'params': image_encoder.parameters()},
                        {'params': text_encoder.parameters()},
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

                    text_batch_size = int(batch_size * cfg.MODALITY.TEXT_BATCH_RATIO)
                    image_batch_size = batch_size - text_batch_size

                    text_loader = None
                    if text_batch_size > 0:
                        text_loader = DataLoader(
                            text_dataset,
                            batch_size=text_batch_size,
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

                    best_val_dict, last_iter_dict = train(
                        logit_head, image_encoder, text_encoder, 
                        image_loader, val_loader, text_loader, 
                        optimizer, scheduler, criterion, iters,
                        eval_freq=cfg.OPTIM.EVAL_FREQ)
                    
                    checkpoint_dir = os.path.join(save_dir, hyperparams_str)
                    makedirs(checkpoint_dir)
                    best_val_path = os.path.join(checkpoint_dir, "best_val.pth")
                    last_iter_path = os.path.join(checkpoint_dir, "last_iter.pth")
                    torch.save(best_val_dict, best_val_path)
                    torch.save(last_iter_dict, last_iter_path)


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
        "--cross-modal-config-file",
        type=str,
        default="",
        help="path to config file for cross-modal training",
    )
    parser.add_argument(
        "--architecture-config-file",
        type=str,
        default="",
        help="path to config file for architecture",
    )
    parser.add_argument(
        "--logit-config-file",
        type=str,
        default="",
        help="path to config file for logit calculation",
    )
    parser.add_argument(
        "--hyperparams-config-file",
        type=str,
        default="",
        help="path to config file for hyperparams",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)