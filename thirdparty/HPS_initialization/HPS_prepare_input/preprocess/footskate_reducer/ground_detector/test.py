# Test on the ground-contact datasets (11 videos)
import argparse
import os
import shutil
import tempfile
from collections import OrderedDict

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import DatasetContact
from net import BasicTemporalModel
from utils import bin_to_bool, read_json

from dataset import REST_INDS


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Test a ground-contact predictor')

    # Data
    parser.add_argument('--root', default='/home/ylzou/research/WACV2020/ground_detector/data/contact_dataset')
    parser.add_argument('--time_window', type=int, default=3, help='How many nearby (past or future) frames to see, length=2*time_window+1')
    parser.add_argument('--pose_norm', type=int, default=1, choices=[0, 1], help='Normalize pose or not')
    parser.add_argument('--data_mode', type=str, default='kp', choices=['op', 'kp'], help='Choose which part of data to use')
    # Model
    parser.add_argument('--num_blocks', type=int, default=2, help='How many residual blocks to use')
    parser.add_argument('--num_features', type=int, default=512, help='Number of channels in the intermediate layers')
    parser.add_argument('--ckpt', default='pretrained/ckpt/model_best.pth', help='Path to save models and logs')
    # Optimization
    parser.add_argument('--bs', type=int, default=512, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='How many workers for data loading')
    # Logging
    parser.add_argument('--out_vid', type=int, default=1, choices=[0, 1], help='Output video or not')
    parser.add_argument('--out_dir', default='cnn_results', help='Path to save models and logs')
    parser.add_argument('--out_prefix', default='')


    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Called with args:')
    print(args)

    ## Dataset
    dataset = DatasetContact(root=args.root, time_window=args.time_window, pose_norm=bin_to_bool(args.pose_norm), 
        output_mode=args.data_mode)

    ## Model
    assert args.time_window > 0
    # Multiple frame
    if args.data_mode == 'op':
        in_channels = len(REST_INDS)*3
    elif args.data_mode == 'kp':
        in_channels = len(REST_INDS)*2
    else:
        raise NotImplementedError

    model = BasicTemporalModel(in_channels=in_channels, num_features=args.num_features, num_blocks=args.num_blocks, 
        time_window=args.time_window)
        
    ckpt = torch.load(args.ckpt)
    state_dict = ckpt['model']
    model.load_state_dict(state_dict)

    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    all_preds = []
    all_labels = []
    results = []
    for v in range(dataset.__vidlen__()):
        num_frames = dataset.__getvid__()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, 
            drop_last=False)
        dataiterator = iter(dataloader)
        steps_per_epoch = int(np.ceil(len(dataset)/args.bs))

        preds = []
        labels = []
        for _ in range(steps_per_epoch):
            inputs = next(dataiterator)
            body, label = inputs
            body = body.cuda()

            with torch.no_grad():
                pred_prob = model(body)
            pred_prob = pred_prob.cpu().numpy()
            pred = pred_prob.copy()
            pred[pred_prob>=0.99] = 1
            pred[pred_prob<0.99] = 0
            preds.append(pred)
            labels.append(label.numpy())
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        all_preds.append(preds)
        all_labels.append(labels)

        # Eval
        TP = np.logical_and(labels==1, preds==1)
        FP = np.logical_and(labels==0, preds==1)
        TN = np.logical_and(labels==0, preds==0)
        FN = np.logical_and(labels==1, preds==0)
        stat_dict = OrderedDict()
        for i, key in enumerate(['All', 'LToe', 'LHeel', 'RToe', 'RHeel']):
            stat_dict[key] = {}
            eps = 1e-6
            if i == 0:
                pre = TP.sum()/(TP.sum()+FP.sum()+eps)
                rec = TP.sum()/(TP.sum()+FN.sum()+eps)
            else:
                pre = TP[:, i-1].sum()/(TP[:, i-1].sum()+FP[:, i-1].sum()+eps)
                rec = TP[:, i-1].sum()/(TP[:, i-1].sum()+FN[:, i-1].sum()+eps)
            f1 = 2.*pre*rec/(pre+rec+eps)
            stat_dict[key]['Precision'] = pre
            stat_dict[key]['Recall'] = rec
            stat_dict[key]['F1'] = f1

        results.append(stat_dict)

        # Visualization
        if bin_to_bool(args.out_vid):
            curr_vid = dataset.vid_list[dataset.vid_idx-1]
            curr_dump = os.path.join(args.out_dir, curr_vid.split('/')[-1][:-4])
            if not os.path.exists(curr_dump):
                os.makedirs(curr_dump)

            vid_cap = cv2.VideoCapture(curr_vid)
            for t in range(num_frames):
                ret, im = vid_cap.read()
                cv2.putText(im, 'Frame: {}/{}'.format(t, num_frames), (50, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
                kp = dataset.pose_2ds[t, :, :]
                feet = kp[[19, 14, 22, 11], :2]
                for i in range(4):
                    if preds[t, i] == 1:
                        color = (0, 255, 0)
                    elif preds[t, i] == 0:
                        color = (0, 0, 255)
                    else:
                        continue
                    cv2.circle(im, (int(feet[i, 0]), int(feet[i, 1])), 5, color, -1)

                out_name = os.path.join(curr_dump, '{:05d}.png'.format(t))
                cv2.imwrite(out_name, im)
            vid_save_name = os.path.join(args.out_dir, curr_vid.split('/')[-1])
            cmd = '/usr/bin/ffmpeg -i "{}/%05d.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -r 50 -pix_fmt yuv420p -y "{}"'.format(curr_dump, vid_save_name)
            os.system(cmd)

    # For all the sequences
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    TP = np.logical_and(all_labels==1, all_preds==1)
    FP = np.logical_and(all_labels==0, all_preds==1)
    TN = np.logical_and(all_labels==0, all_preds==0)
    FN = np.logical_and(all_labels==1, all_preds==0)
    stat_dict = OrderedDict()
    for i, key in enumerate(['All', 'LToe', 'LHeel', 'RToe', 'RHeel']):
        stat_dict[key] = {}
        eps = 1e-6
        if i == 0:
            pre = TP.sum()/(TP.sum()+FP.sum()+eps)
            rec = TP.sum()/(TP.sum()+FN.sum()+eps)
        else:
            pre = TP[:, i-1].sum()/(TP[:, i-1].sum()+FP[:, i-1].sum()+eps)
            rec = TP[:, i-1].sum()/(TP[:, i-1].sum()+FN[:, i-1].sum()+eps)
        f1 = 2.*pre*rec/(pre+rec+eps)
        stat_dict[key]['Precision'] = pre
        stat_dict[key]['Recall'] = rec
        stat_dict[key]['F1'] = f1
    results.append(stat_dict) 

    vid_list = dataset.vid_list+['All']
    for v, vid in enumerate(vid_list):
        print(vid.split('/')[-1])
        print(results[v])
        print('\n')


if __name__ == '__main__':
    main()
