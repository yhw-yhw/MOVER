import argparse

from pare.eft.eft_fitter import EFTFitter
from pare.eft.config import get_cfg_defaults


def main(args):
    hparams = get_cfg_defaults()
    hparams.merge_from_list(args.opts)

    eft_fitter = EFTFitter(hparams)

    eft_fitter.finetune()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opts', default=[], nargs='*', help='additional options to update config')
    args = parser.parse_args()

    main(args)