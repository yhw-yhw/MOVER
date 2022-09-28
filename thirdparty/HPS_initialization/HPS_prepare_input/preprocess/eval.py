import torch
import argparse
from loguru import logger
import pytorch_lightning as pl

from pare.utils.os_utils import copy_code
from pare.utils.eval_utils import find_best_ckpt
from pare.core.single_image_trainer import SingleImageTrainer
from pare.utils.train_utils import load_pretrained_model
from pare.core.config import run_grid_search_experiments


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')

    model = SingleImageTrainer(hparams=hparams).to(device)

    if hparams.TRAINING.PRETRAINED_LIT is not None:
        logger.warning(f'Loading pretrained model from {hparams.TRAINING.PRETRAINED_LIT}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED_LIT)['state_dict']
        load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True)

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(
        gpus=1,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
        logger=None,
    )

    logger.info('*** Started testing ***')
    trainer.test(model=model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--opts', default=[], nargs='*', help='additional options to update config')
    parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')
    parser.add_argument('--cluster', default=False, action='store_true', help='creates submission files for cluster')
    parser.add_argument('--bid', type=int, default=5, help='amount of bid for cluster')
    parser.add_argument('--memory', type=int, default=32000, help='memory amount for cluster')
    parser.add_argument('--num_cpus', type=int, default=8, help='num cpus for cluster')
    parser.add_argument('--gpu_min_mem', type=int, default=10000, help='minimum amount of GPU memory')
    parser.add_argument('--gpu_arch', default=['tesla', 'quadro', 'rtx'],
                        nargs='*', help='additional options to update config')
    parser.add_argument('--no_best_ckpt', action='store_true')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    if not 'TRAINING.PRETRAINED_LIT' in args.opts and not args.no_best_ckpt:
        if '05.11-spin_ckpt_eval' in args.cfg:
            pass
        else:
            ckpt_path = find_best_ckpt(args.cfg, new_version=True)
            args.opts += ['TRAINING.PRETRAINED_LIT', ckpt_path]

    args.opts += ['TRAINING.RESUME', None]

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        bid=args.bid,
        use_cluster=args.cluster,
        memory=args.memory,
        script='eval.py',
        cmd_opts=args.opts,
        gpu_min_mem=args.gpu_min_mem,
        gpu_arch=args.gpu_arch,
    )

    hparams.RUN_TEST = True

    main(hparams)
