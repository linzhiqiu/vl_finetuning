# Vision&Language Finetuning

# Prompt Learning for Vision-Language Models

This repo contains the code for adapting vision-language models like [CLIP](https://arxiv.org/abs/2103.00020) to downstream datasets. The arxiv preprint is at:

<!-- * [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557), in CVPR, 2022.
* [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134), IJCV, 2022. -->

<!-- ## Updates

- **16.07.2022**: CoOp has been accepted to IJCV for publication! -->

## How to Install
We recommend to install the environment through conda and pip. You should make a new environment with python>=3.9, for example:

```
conda create -n vl_finetuning python=3.9
```

Next, you can download pytorch from official site, for example:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Next, run `pip install -r requirements.txt` in this repo to install a few more packages required by [CLIP](https://github.com/openai/CLIP). You don't need to install `dassl`.

Follow [DATASETS.md](DATASETS.md) to install the downstream datasets. Note that we use the original split of data (including the few-shot splits for seed 1-3, except for ImageNet) to ensure a fair comparison to [CoOp](https://github.com/KaiyangZhou/CoOp).

## Configuration
Similar to [CoOp](https://github.com/KaiyangZhou/CoOp), we use [yacs](https://github.com/rbgirshick/yacs) to specify all experiment configurations for reproducibility. The root configuration file can be found at [engine/config/default.py](engine/config/default.py), and you may want to modify the `_C.DATA_DIR` path to where you install all the datasets. Note that this config is different from the default one in CoOp codebase; for simplicity, we remove irrelevant configuration for semi-supervised learning and only keep the ones for vision&language finetuning.

## Convert few-shot train/val sets from CoOp pickle objects to json

```
python convert_pickle_to_index.py --dataset-config-file config/datasets/oxford_pets.yaml --few-shot-config-file config/few_shot/shot_1.yaml SEED 1
```

## Split few-shot train/val sets
We provide few-shot train/val splits for seed 1, 2, 3, and shots 1, 2, 4, 8, 16 in [indices/](indices/), as they were generated from the original [CoOp codebase](https://github.com/KaiyangZhou/CoOp) (except for ImageNet). If you want to generate more splits with different shots and seeds, please refer to [split_for_few_shot.py]. You will need to specify a dataset config yaml file such as [engine/config/datasets/imagenet.yaml](configs/datasets/imagenet.yaml), and a few-shot config yaml file such as [engine/configs/few_shot/shot_16.yaml](configs/few_shot/shot_16.yaml). Then run:

```
python split_for_few_shot.py --dataset-config-file config/datasets/imagenet.yaml --few-shot-config-file config/few_shot/shot_1.yaml SEED 1
```

## Feature Extraction
We provide few-shot train/val splits for seed 1, 2, 3, and shots 1, 2, 4, 8, 16 in [indices/](indices/), as they were generated from the original [CoOp codebase](https://github.com/KaiyangZhou/CoOp) (except for ImageNet). If you want to generate more splits with different shots and seeds, please refer to [split_for_few_shot.py]. You will need to specify a dataset config yaml file such as [engine/config/datasets/imagenet.yaml](configs/datasets/imagenet.yaml), and a few-shot config yaml file such as [engine/configs/few_shot/shot_16.yaml](configs/few_shot/shot_16.yaml). Then run:

```
python split_for_few_shot.py --dataset-config-file config/datasets/imagenet.yaml --few-shot-config-file config/few_shot/shot_1.yaml SEED 1
```

<!-- ## How to Run

Click a paper below to see the detailed instructions on how to run the code to reproduce the results.

* [Learning to Prompt for Vision-Language Models](COOP.md)
* [Conditional Prompt Learning for Vision-Language Models](COCOOP.md) -->

<!-- ## Models and Results

- The pre-trained weights of CoOp (both M=16 & M=4) on ImageNet based on RN50, RN101, ViT-B/16 and ViT-B/32 can be downloaded altogether via this [link](https://drive.google.com/file/d/18ypxfd82RR0pizc5MM1ZWDYDk4j0BtPF/view?usp=sharing). The weights can be used to reproduce the results in Table 1 of CoOp's paper (i.e., the results on ImageNet and its four variants with domain shift). To load the weights and run the evaluation code, you will need to specify `--model-dir` and `--load-epoch` (see this [script](https://github.com/KaiyangZhou/CoOp/blob/main/scripts/eval.sh) for example).
- The raw numerical results can be found at this [google drive link](https://docs.google.com/spreadsheets/d/12_kaFdD0nct9aUIrDoreY0qDunQ9q9tv/edit?usp=sharing&ouid=100312610418109826457&rtpof=true&sd=true). -->

## Citation
If you use this code in your research, please kindly cite the following papers

<!-- ```bash
@inproceedings{zhou2022cocoop,
    title={Conditional Prompt Learning for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
``` -->