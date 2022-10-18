import argparse
# import wandb
import os
import math
import copy
import json

import torch
import torch.nn as nn
import torch.multiprocessing
import torchvision
torch.multiprocessing.set_sharing_strategy('file_system')
import random
import numpy as np
import clip

torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)

from modeling import load_model
from data import load_dataset, get_classnames
# from engine import train
from configs import get_default_cfg

"""optimizers.py Contains the optimizers that are supported by our finetuning framework. This was taken from https://github.com/KaiyangZhou/Dassl.pytorch, so we thank them for the original code.
"""

import warnings
import torch
import torch.nn as nn

AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "adamw"]


def build_optimizer(model, optim_cfg, param_groups=None):
    """A function wrapper for building an optimizer.
    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
        param_groups: If provided, directly optimize param_groups and abandon model
    """

    # print(f"Building optimizer: {optim_cfg.NAME}")

    optim = optim_cfg.NAME
    lr = optim_cfg.BASE_LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPENING
    sgd_nesterov = optim_cfg.SGD_NESTEROV
    rmsprop_alpha = optim_cfg.RMSPROP_ALPHA
    adam_beta1 = optim_cfg.ADAM_BETA1
    adam_beta2 = optim_cfg.ADAM_BETA2
    staged_lr = optim_cfg.STAGED_LR
    new_layers = optim_cfg.NEW_LAYERS
    base_lr_mult = optim_cfg.BASE_LR_MULT

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            f"optim must be one of {AVAI_OPTIMS}, but got {optim}"
        )

    if param_groups is not None and staged_lr:
        warnings.warn(
            "staged_lr will be ignored, if you need to use staged_lr, "
            "please bind it with param_groups yourself."
        )

    if param_groups is None:
        if staged_lr:
            if not isinstance(model, nn.Module):
                raise TypeError(
                    "When staged_lr is True, model given to "
                    "build_optimizer() must be an instance of nn.Module"
                )

            if isinstance(model, nn.DataParallel):
                model = model.module

            if isinstance(new_layers, str):
                if new_layers is None:
                    warnings.warn("new_layers is empty (staged_lr is useless)")
                new_layers = [new_layers]

            base_params = []
            base_layers = []
            new_params = []

            for name, module in model.named_children():
                if name in new_layers:
                    new_params += [p for p in module.parameters()]
                else:
                    base_params += [p for p in module.parameters()]
                    base_layers.append(name)

            param_groups = [
                {
                    "params": base_params,
                    "lr": lr * base_lr_mult
                },
                {
                    "params": new_params
                },
            ]

        else:
            if isinstance(model, nn.Module):
                param_groups = model.parameters()
            else:
                param_groups = model

    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    else:
        raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

    return optimizer

"""schedulers.py Contains the learning rate schedulers that are supported by our finetuning framework. This was taken from https://github.com/KaiyangZhou/Dassl.pytorch, so we thank them for the original code.
"""

from torch.optim.lr_scheduler import _LRScheduler

AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]


def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    # print(f"Building scheduler: {optim_cfg.LR_SCHEDULER}")
    lr_scheduler = optim_cfg.LR_SCHEDULER
    stepsize = optim_cfg.STEPSIZE
    gamma = optim_cfg.GAMMA
    max_epoch = int(optim_cfg.NUM_EPOCHS * optim_cfg.EPOCH_MULT)

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                f"be an integer, but got {type(stepsize)}"
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                f"be a list, but got {type(stepsize)}"
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

    if optim_cfg.WARMUP_EPOCHS > 0:
        if not optim_cfg.WARMUP_RECOUNT:
            scheduler.last_epoch = optim_cfg.WARMUP_EPOCHS

        if optim_cfg.WARMUP_TYPE == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCHS,
                optim_cfg.WARMUP_CONS_LR
            )

        elif optim_cfg.WARMUP_TYPE == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCHS,
                optim_cfg.WARMUP_MIN_LR
            )

        else:
            raise ValueError

    return scheduler

def merge_from_args(cfg, args):
    if args.verbose:
        print("LR:", args.lr)
    cfg.MODEL.BACKBONE = args.model
    cfg.DATASET.NUM_SHOTS = args.shots
    cfg.OPTIM.BASE_LR = args.lr
    cfg.OPTIM.WEIGHT_DECAY = args.wd
    cfg.OPTIM.NUM_EPOCHS = args.iters
    cfg.OPTIM.WARMUP_EPOCHS = args.warmup
    cfg.OPTIM.CLIP_GRAD = args.clip_grad
    cfg.OPTIM.NAME = args.optim

    model_name = args.model.replace('/', '-')
    model_name += "_noproj" if args.no_proj else ""

    name = f"{model_name}_{args.dataset}_{args.optim}_{args.shots}_{args.load_linear}_{args.views}_{args.normalize}_{args.text_ratio}_{args.bias}_{args.text_source}_{not args.no_eval}_{args.val_views}_{args.model_type}_{args.seed}_{args.train_seed}"

    cfg.NAME = name

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)

def make_features(dataloader, num_views, clip_model, cfg, mode, args):
    print("Making features for", mode)
    all_labels = {}
    all_features = {}    
    clip_model = ImageCLIP(clip_model)
    clip_model = nn.DataParallel(clip_model)

    for i in range(num_views):
        print(f"{i}/{num_views}")
        all_labels[i] = []
        all_features[i] = []
        with torch.no_grad():
            for batch in dataloader:
                try:
                    images = batch["images"].to(cfg.MODEL.DEVICE)
                    labels = batch["labels"]
                except:
                    images, labels = batch
                    images = images.to(cfg.MODEL.DEVICE)

                img_features = clip_model(images).detach().cpu().float()
                all_features[i].append(img_features)
                all_labels[i] += labels

        all_labels[i] = torch.tensor(all_labels[i])
        all_features[i] = torch.cat(all_features[i], dim=0)


    os.makedirs(os.path.join(args.root, f"dataset_features/{cfg.MODEL_NAME}/{args.dataset}") , exist_ok=True)
    if mode == "train":
        torch.save({"features": all_features, "labels": all_labels}, os.path.join(args.root, f"dataset_features/{cfg.MODEL_NAME}/{args.dataset}/{mode}_features_{args.shots}_{args.seed}_{num_views}.pt"))
    elif mode == "val":
        shots = min(args.shots, 4)
        torch.save({"features": all_features, "labels": all_labels}, os.path.join(args.root, f"dataset_features/{cfg.MODEL_NAME}/{args.dataset}/{mode}_features_{shots}_{args.seed}_{num_views}.pt"))
    else:
        torch.save({"features": all_features, "labels": all_labels}, os.path.join(args.root, f"dataset_features/{cfg.MODEL_NAME}/{args.dataset}/test_features1.pt"))

    return all_features, all_labels


def make_model(args, output_dim, num_cls):
    if args.model_type == "linear":
        model = nn.Linear(output_dim, num_cls, bias=args.bias)
    elif args.model_type == "mlp":
        model = torchvision.ops.MLP(output_dim, [1024, 1024, num_cls])
    return model

def eval(model, all_features, all_labels, cfg, loss_fn, args):
    val_correct = 0
    val_loss = 0
    total = 0
    with torch.no_grad():
        for j in range(len(all_features)):
            features, labels = all_features[j], all_labels[j]
            for i in range(0, len(labels), args.batch_size):
                batch_features = features[i:i+args.batch_size].to(cfg.MODEL.DEVICE)
                batch_labels = labels[i:i+args.batch_size].to(cfg.MODEL.DEVICE)

                logits = model(batch_features)
                loss = loss_fn(logits, batch_labels)
                val_correct += torch.sum(torch.argmax(logits, dim=1) == batch_labels).item()
                val_loss += loss.item()
                total += len(batch_labels)

    return val_correct/total, val_loss/total

def iterate(model, all_features, all_labels, loss_fn, optimizer, scheduler, order, cfg, text_features, text_labels, text_order, args, ival):
    text_batch_size = int(args.batch_size * args.text_ratio)
    image_batch_size = args.batch_size - text_batch_size

    if len(order) < image_batch_size:
        order = np.random.permutation(len(all_labels[0]))
        ival = random.choice(range(len(all_labels)))
    features, labels = all_features[ival], all_labels[ival]

    batch_features = features[order[:image_batch_size]].to(cfg.MODEL.DEVICE)
    batch_labels = labels[order[:image_batch_size]].to(cfg.MODEL.DEVICE)
    if args.text_comb_ratio > 0:
        text_features = text_features[batch_labels].to(cfg.MODEL.DEVICE)
        r = random.random()/2
        batch_features = batch_features * (1 - r) + text_features * r
    elif text_batch_size > 0:
        if len(text_order) < text_batch_size:
            text_order = np.random.permutation(len(text_labels))
        text_batch_features = text_features[text_order[:text_batch_size]].to(cfg.MODEL.DEVICE)
        text_batch_labels = text_labels[text_order[:text_batch_size]].to(cfg.MODEL.DEVICE)
        
        batch_features = torch.cat((batch_features, text_batch_features), dim=0)
        batch_labels = torch.cat((batch_labels, text_batch_labels), dim=0)

    optimizer.zero_grad()
    logits = model(batch_features)
    loss = loss_fn(logits, batch_labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    iter_correct = torch.sum(torch.argmax(logits, dim=1) == batch_labels).item()
    return loss.item(), iter_correct, order[image_batch_size:], text_order[text_batch_size:], len(batch_labels), ival
    
def combine(old, trained, ratio, args):
    new = copy.deepcopy(trained)
    new.weight = nn.Parameter(new.weight * ratio + old.clone() * (1 - ratio))
    if args.bias:
        new.bias = nn.Parameter(new.bias * ratio)
    return new

def train(linear_classifier, train_features, train_labels, val_features, val_labels, val_train_features, val_train_labels, test_features, test_labels, text_features, text_labels, optimizer, loss_fn, scheduler, cfg, main_results, args):
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    os.makedirs(os.path.join(args.root, f"{cfg.NAME}"), exist_ok=True)
    string_name = f"LR{args.lr}_WD{args.wd}_ITERS{args.iters}_{args.batch_size}"

    order = np.random.permutation(len(train_labels[0]))
    if args.text_ratio > 0:
        text_order = np.random.permutation(len(text_labels))
    else:
        text_order = []
    
    total_loss, train_correct, total_count, best_val_acc = 0, 0, 0, 0
    best_val = None
    train_accs, val_accs, train_losses, val_losses = [0], [0], [99999], [99999]

    ival = 0
    for iter in range(args.iters):
        iter_loss, iter_correct, order, text_order, bsize, ival = iterate(linear_classifier, train_features, train_labels, loss_fn, optimizer, scheduler, order, cfg, text_features, text_labels, text_order, args, ival)

        train_accs.append(iter_correct / bsize)
        train_losses.append(iter_loss / bsize)

        total_loss += iter_loss
        train_correct += iter_correct
        total_count += bsize

        if iter % args.eval_every == 0 and (not args.no_eval):
            val_acc, val_loss = eval(linear_classifier, val_train_features, val_train_labels, cfg, loss_fn, args)

            if args.verbose:
                print(f"Train Loss: {total_loss / total_count}, Train Accuracy: {total_correct / total_count}")
                print(f"Val Loss: {val_loss}, Val Accuracy: {val_acc}")
            total_loss, total_correct, total_count = 0, 0, 0
            
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val = copy.deepcopy(linear_classifier)
                torch.save(linear_classifier.state_dict(), os.path.join(args.root, f"{cfg.NAME}/{string_name}.pt"))
    
    if args.no_eval:
        torch.save(linear_classifier.state_dict(), os.path.join(args.root, f"{cfg.NAME}/{string_name}.pt"))
        
    results["final_val_loss"] = val_losses[-1]
    results["final_val_acc"] = val_accs[-1]
    results["final_train_loss"] = train_losses[-1]
    results["final_train_acc"] = train_accs[-1]
    results["best_val_acc"] = best_val_acc

    print(f"Final Train Loss: {train_losses[-1]}, Final Train Accuracy: {train_accs[-1]}")
    print(f"Final Val Loss: {val_losses[-1]}, Final Val Accuracy: {val_accs[-1]}")
    print(f"Best Val Accuracy: {best_val_acc}")
        
    with torch.no_grad():
        if best_val is not None and len(val_labels[0]) > args.min_val * args.val_views:
            linear_classifier = best_val
        test_acc, test_loss = eval(linear_classifier, test_features, test_labels, cfg, loss_fn, args)

        print(f"Test Loss (best): {test_loss}, Test Accuracy (best): {test_acc}")
        results["test_loss_val"] = test_loss
        results["test_acc_val"] = test_acc

        if args.model_type == "linear":
            old_weights = torch.load(os.path.join(cfg.FEATURE_DIR, "weights", cfg.MODEL.BACKBONE.replace('/', '-'), f"{cfg.DATASET.NAME}_frozen_text_weights.pt"))
            wise_accs = []
            ratios = [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            for alpha in ratios:
                new_classifier = combine(old_weights, linear_classifier, alpha, args)
                wise_acc, _ = eval(new_classifier, val_features, val_labels, cfg, loss_fn, args)
                wise_accs.append(wise_acc)

            best_alpha = ratios[wise_accs.index(max(wise_accs))]
            new_classifier = combine(old_weights, linear_classifier, best_alpha, args)
            wise_acc, _ = eval(new_classifier, test_features, test_labels, cfg, loss_fn, args)

            print(f"Wise Accuracy: {wise_acc}, Ratio: {best_alpha}")
            results["wise_acc"] = wise_acc
            results["wise_val_acc"] = max(wise_accs)
            results["wise_alpha"] = best_alpha
            results["all_wise"] = wise_accs

            torch.save(new_classifier.state_dict(), os.path.join(args.root, f"{cfg.NAME}/{string_name}_wise.pt"))
        else:
            results["wise_acc"] = 0
            results["wise_val_acc"] =0
            results["wise_alpha"] = 0
            results["all_wise"] = 0

    with open(os.path.join(args.root, f"{cfg.NAME}/results.json"), "w") as f:
        main_results[string_name] = results
        json.dump(main_results, f)    

    return results

def full_loop(args):
    cfg = get_default_cfg()
    cfg.merge_from_file(f"configs/datasets/{args.dataset}.yaml")

    merge_from_args(cfg, args)
    print("Dataset:", args.dataset)

    model_name = args.model.replace('/', '-')
    model_name += "_noproj" if args.no_proj else ""
    cfg.MODEL_NAME = model_name

    string_name = f"LR{args.lr}_WD{args.wd}_ITERS{args.iters}_{args.batch_size}"
    if not os.path.exists(os.path.join(args.root, f"{cfg.NAME}/results.json")):
        print("Creating results.json")
        main_results = {}
    else:
        print("Loading results.json from", os.path.join(args.root, f"{cfg.NAME}/results.json"))
        with open(os.path.join(args.root, f"{cfg.NAME}/results.json"), "r") as f:
            main_results = json.load(f)
        if string_name in main_results:
            if not args.redo: 
                return main_results[string_name]
        else:
            print(string_name)

    clip_model, _ = clip.load(args.model, device=cfg.MODEL.DEVICE)
    if args.no_proj:
        try:
            clip_model.visual.proj = nn.Identity()
        except:
            pass
    if args.views == 1:
        cfg.DATA.TRANSFORMS = "no_aug"

    train_dataloader, val_dataloader, test_dataloader = None, None, None
    if args.val_views > 1:
        cfg.DATA.VAL_AUGMENT = True
    ROOT = os.path.join(args.root, f"dataset_features/{cfg.MODEL_NAME}/{args.dataset}/")
    print(ROOT)
    if os.path.exists(ROOT + f"train_features_{args.shots}_{args.seed}_{args.views}.pt"):
        print("Loading features", ROOT + f"train_features_{args.shots}_{args.seed}_{args.views}.pt")
        tf = torch.load(f"{ROOT}train_features_{args.shots}_{args.seed}_{args.views}.pt")
        train_features, train_labels = tf["features"], tf["labels"]
    else:
        print("Making features", ROOT + f"train_features_{args.shots}_{args.seed}_{args.views}.pt")
        train_dataloader, val_dataloader, test_dataloader = load_dataset(cfg)
        train_features, train_labels = make_features(train_dataloader, args.views, clip_model, cfg, "train", args)
    
    if args.text_ratio > 0 or args.text_comb_ratio > 0:
        assert not (args.text_comb_ratio > 0 and args.text_source not in ["basic", "openai"])
        if args.text_comb_ratio > 0:
            args.text_ratio = 0
        text_data = torch.load(f"/data3/samuelyu/finetuning/features/text_features/{cfg.MODEL_NAME}/{args.text_source}/{args.dataset}.pt")
        text_features, text_labels = text_data["features"], text_data["labels"]
    else:
        text_features, text_labels = None, None
    
    val_shots = min(args.shots, 4)
    if os.path.exists(ROOT + f"val_features_{val_shots}_{args.seed}_{args.val_views}.pt"):
        print("Loading features", ROOT + f"val_features_{val_shots}_{args.seed}_{args.val_views}.pt")
        vf = torch.load(ROOT + f"val_features_{val_shots}_{args.seed}_{args.val_views}.pt")
        val_train_features, val_train_labels = vf["features"], vf["labels"]
    else:
        if val_dataloader is None:
            train_dataloader, val_dataloader, test_dataloader = load_dataset(cfg)
        val_train_features, val_train_labels = make_features(val_dataloader, args.val_views, clip_model, cfg, "val", args)

    if args.val_views != 1:
        if os.path.exists(ROOT + f"val_features_{val_shots}_{args.seed}_1.pt"):
            print("Loading features", ROOT + f"val_features_{val_shots}_{args.seed}_1.pt")
            vf = torch.load(ROOT + f"val_features_{val_shots}_{args.seed}_1.pt")
            val_features, val_labels = vf["features"], vf["labels"]
        else:
            cfg.DATA.TRANSFORMS = "no_aug"
            cfg.DATA.VAL_AUGMENT = False
            train_dataloader, val_dataloader, test_dataloader = load_dataset(cfg)
            val_features, val_labels = make_features(val_dataloader, 1, clip_model, cfg, "val", args)
    else:
        val_features, val_labels = val_train_features, val_train_labels
    

    if os.path.exists(ROOT + f"test_features1.pt"):
        print("Loading features", ROOT + f"test_features1.pt")
        testf = torch.load(ROOT + "test_features1.pt")
        test_features, test_labels = testf["features"], testf["labels"]
    else:
        if test_dataloader is None:
            train_dataloader, val_dataloader, test_dataloader = load_dataset(cfg)
        test_features, test_labels = make_features(test_dataloader, 1, clip_model, cfg, "test", args)

    if args.normalize:
        for i in train_features:
            train_features[i] = train_features[i] / train_features[i].norm(dim=-1, keepdim=True)
        for i in val_features:
            val_features[i] = val_features[i] / val_features[i].norm(dim=-1, keepdim=True)
        for i in test_features:
            test_features[i] = test_features[i] / test_features[i].norm(dim=-1, keepdim=True)
        if text_features is not None:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    classnames = get_classnames(cfg.DATASET.NAME)

    torch.manual_seed(args.train_seed)
    random.seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)
    np.random.seed(args.train_seed)

    OUTPUT_DIM = train_features[0].shape[1]
    if args.load_linear:
        weights = torch.load(f"/data3/samuelyu/finetuning/features/text_features/{cfg.MODEL_NAME}/{args.text_source}/{args.dataset}.pt")['features'].clone().cpu()
        linear_classifier = nn.Linear(OUTPUT_DIM, len(classnames), bias=args.bias)
        linear_classifier.weight = nn.Parameter(weights.clone())
    else:
        linear_classifier = make_model(args, OUTPUT_DIM, len(classnames))
    linear_classifier = linear_classifier.to(cfg.MODEL.DEVICE)

    optimizer = build_optimizer(linear_classifier, cfg.OPTIM)
    scheduler = build_lr_scheduler(optimizer, cfg.OPTIM)
    loss_fn = get_loss_fn(cfg)

    results = train(linear_classifier, train_features, train_labels, val_features, val_labels, val_train_features, val_train_labels, test_features, test_labels, text_features, text_labels, optimizer, loss_fn, scheduler, cfg, main_results, args)

    return results

def get_parser():
    parser = argparse.ArgumentParser(description="Finetune a CLIP model on a classification task")
    parser.add_argument("--root", type=str, default="/data3/samuelyu/linear_finetuning")
    parser.add_argument("--model", type=str, default="RN50")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_type", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=12800)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--val_views", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--min_val", type=int, default=128)
    parser.add_argument("--clip_grad", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--optim", type=str, default='adamw')
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--load_linear", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--no_proj", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--text_ratio", type=float, default=0.0)
    parser.add_argument("--text_source", type=str, default="basic")
    parser.add_argument("--text_comb_ratio", type=float, default=0.0)
    parser.add_argument("--redo", action="store_true", default=False)

    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    full_loop(args)