from yacs.config import CfgNode as CN

##### CONFIGS #####
hparams = CN()
hparams.EXEMPLAR_LR = 5e-6
hparams.PRETRAINED_CKPT = 'logs/spin/30.08-eft_dataset_pretrained_mpii3d_fix/01-09-2020_21-16-11_30.08-eft_dataset_pretrained_mpii3d_fix/tb_logs_pare/0_af6934cb2bdf49bc892a61c0306526a9/checkpoints/epoch=77.ckpt'
hparams.MIN_EXEMPLAR_ITER = 20
hparams.MAX_EXEMPLAR_ITER = 20
hparams.MIN_LOSS = 0.0
hparams.SAVE_IMAGES = True
hparams.LOG_DIR = 'logs/eft'
hparams.EXP_NAME = 'default'
hparams.LOG = True
hparams.EVALUATE = True

hparams.LOSS = CN()
hparams.LOSS.LOSS_WEIGHT = 60.
hparams.LOSS.GT_TRAIN_WEIGHT = 1.
hparams.LOSS.BETA_LOSS_WEIGHT = 0.001
hparams.LOSS.KEYPOINT_LOSS_WEIGHT = 5.
hparams.LOSS.OPENPOSE_TRAIN_WEIGHT = 0.
hparams.LOSS.LEG_ORIENTATION_LOSS_WEIGHT = 0.005

hparams.DATASET = CN()
hparams.DATASET.LOAD_TYPE = 'Base'

hparams.DATASET.BATCH_SIZE = 1
hparams.DATASET.NUM_WORKERS = 8
hparams.DATASET.PIN_MEMORY = True
hparams.DATASET.VAL_DS = 'mpi-inf-3dhp'
hparams.DATASET.NUM_IMAGES = -1
hparams.DATASET.IMG_RES = 224
hparams.DATASET.RENDER_RES = 480
hparams.DATASET.FOCAL_LENGTH = 5000.

hparams.SPIN = CN()
hparams.SPIN.BACKBONE = 'resnet50'

def get_cfg_defaults():
    return hparams.clone()