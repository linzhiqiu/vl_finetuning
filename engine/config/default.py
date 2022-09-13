from yacs.config import CfgNode as CN

_C = CN()

###########################
# Directory Config (modify if using your own paths)
###########################
# Directory with installed datasets
_C.DATA_DIR = "./data"
# Directory with few-shot indices
_C.FEW_SHOT_DIR = "./indices"
# Directory to save the extracted features/layers (feature.py)
_C.FEATURE_DIR = f"./features"
# Directory to save the trained linear/non-linear models (logreg_minibatch.py)
_C.LOGREG_MINIBATCH_DIR = f"./logreg_minibatch"
# Directory to save the trained linear/non-linear models (logreg_fullbatch.py)
_C.LOGREG_FULLBATCH_DIR = f"./logreg_fullbatch"
# Directory to save the evaluation results (eval.py)
_C.EVAL_DIR = f"./eval"


###########################
# Seed (must be set for reproducibility)
###########################
_C.SEED = 1

###########################
# Dataset Config
###########################
_C.DATASET = CN()
# Dataset name
_C.DATASET.NAME = ""
# Number of images per class
_C.DATASET.NUM_SHOTS = 16
# Maximum size of val shots (otherwise same size as train shots)
_C.DATASET.MAX_VAL_SHOTS = 4

###########################
# (Image) Feature Extraction Config 
# (need to specify transform in _C.INPUT)
###########################
_C.FEATURE = CN()
# Number of views per train image
_C.FEATURE.VIEWS_PER_TRAIN = 1
# Number of views per val image
_C.FEATURE.VIEWS_PER_VAL = 1
# Which layer to extract features from. 0 if extracting from last layer output
_C.FEATURE.LAYER_IDX = 0

# Image Backbone for CLIP
_C.FEATURE.BACKBONE = ""

###########################
# (Text) Feature Extraction Config 
###########################
_C.TEXT_FEATURE = CN()
# Which layer to extract features from. 0 if extracting from last layer output
_C.TEXT_FEATURE.LAYER_IDX = 0
# Templates to use (defined in engine/template/default.py)
_C.TEXT_FEATURE.TEMPLATE = "default"

###########################
# System definition
###########################

_C.USE_CUDA = True
# Print detailed information
# E.g. trainer, dataset, and backbone
_C.VERBOSE = True

###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
# Mode of interpolation in resize functions
_C.INPUT.INTERPOLATION = "bicubic"
# For available choices please refer to transforms.py
_C.INPUT.TRANSFORMS = ()
# If True, tfm_train and tfm_test will be None
_C.INPUT.NO_TRANSFORM = False
# Mean and std (default: ImageNet)
# _C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# _C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Mean and std (default: CoOp)
_C.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
_C.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
# Random crop
_C.INPUT.CROP_PADDING = 4
# Random resized crop
_C.INPUT.RRCROP_SCALE = (0.08, 1.0)
# Cutout
_C.INPUT.CUTOUT_N = 1
_C.INPUT.CUTOUT_LEN = 16
# Gaussian noise
_C.INPUT.GN_MEAN = 0.0
_C.INPUT.GN_STD = 0.15
# RandomAugment
_C.INPUT.RANDAUGMENT_N = 2
_C.INPUT.RANDAUGMENT_M = 10
# ColorJitter (brightness, contrast, saturation, hue)
_C.INPUT.COLORJITTER_B = 0.4
_C.INPUT.COLORJITTER_C = 0.4
_C.INPUT.COLORJITTER_S = 0.4
_C.INPUT.COLORJITTER_H = 0.1
# Random gray scale's probability
_C.INPUT.RGS_P = 0.2
# Gaussian blur
_C.INPUT.GB_P = 0.5  # propability of applying this operation
_C.INPUT.GB_K = 21  # kernel size (should be an odd number)

###########################
# Dataloader (For feature extraction)
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.TEST_BATCH_SIZE = 32

###########################
# Architecture (for mini-batch logistic regression)
###########################
_C.ARCHITECTURE = CN()
_C.ARCHITECTURE.HEAD = "linear"
_C.ARCHITECTURE.BIAS = False

###########################
# Modality (for mini-batch logistic regression)
###########################
_C.MODALITY = CN()
_C.MODALITY.TEXT_BATCH_RATIO = 0.5

###########################
# Logit calculation (during mini-batch logistic regression training)
###########################
_C.LOGIT = CN()
_C.LOGIT.FEATURE_NORM = False
_C.LOGIT.HEAD_NORM = False
_C.LOGIT.USE_LOGIT_SCALE = False

###########################
# Optimization (for mini-batch logistic regression)
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = "adamw"
# Total iter is MAX_ITER + WARMUP_ITER
_C.OPTIM.MAX_ITER = [12800]
_C.OPTIM.BATCH_SIZE = [32]
_C.OPTIM.LR = [0.0003]
_C.OPTIM.WEIGHT_DECAY = [0.0]
_C.OPTIM.MOMENTUM = 0.9
# _C.OPTIM.SGD_DAMPNING = 0
_C.OPTIM.SGD_NESTEROV = False
# The following also apply to other
# adaptive optimizers like adamw
_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.999
# # STAGED_LR allows different layers to have
# # different lr, e.g. pre-trained base layers
# # can be assigned a smaller lr than the new
# # classification layer
# _C.OPTIM.STAGED_LR = False
# _C.OPTIM.NEW_LAYERS = ()
# _C.OPTIM.BASE_LR_MULT = 0.1
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = "cosine"
# Set WARMUP_ITER larger than 0 to activate warmup training
_C.OPTIM.WARMUP_ITER = 0
# Either linear or constant
_C.OPTIM.WARMUP_TYPE = "linear"
# WARMUP learning rate when type=linear/constant
_C.OPTIM.WARMUP_MIN_LR = 1e-5
# Evaluation on val set every EVAL_FREQ iters
_C.OPTIM.EVAL_FREQ = 100


###########################
# Logistic Regression (full-batch)
###########################
_C.LOGREG_FULLBATCH = CN()
_C.LOGREG_FULLBATCH.SEARCH_LIST = []
_C.LOGREG_FULLBATCH.SEEDS = []
_C.LOGREG_FULLBATCH.MAX_ITER = 1000
_C.LOGREG_FULLBATCH.NUM_STEP = 8

# ###########################
# # Test
# ###########################
# _C.TEST = CN()
# _C.TEST.EVALUATOR = "Classification"
# _C.TEST.PER_CLASS_RESULT = False
# # Compute confusion matrix, which will be saved
# # to $OUTPUT_DIR/cmat.pt
# _C.TEST.COMPUTE_CMAT = False
# # If NO_TEST=True, no testing will be conducted
# _C.TEST.NO_TEST = False
# # Use test or val set for FINAL evaluation
# _C.TEST.SPLIT = "test"
# # Which model to test after training (last_step or best_val)
# # If best_val, evaluation is done every epoch (if val data
# # is unavailable, test data will be used)
# _C.TEST.FINAL_MODEL = "last_step"


def get_cfg_default():
    return _C.clone()
