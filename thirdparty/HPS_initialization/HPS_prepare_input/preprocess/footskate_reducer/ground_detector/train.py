import argparse
import os
from collections import OrderedDict

import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import DatasetHuman36m, DatasetMads, DatasetConcat
from net import BasicTemporalModel
from utils import save_ckpt, evaluate, bin_to_bool, balanced_loss

from dataset import REST_INDS

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Training a ground-contact predictor')

    # Data
    parser.add_argument('--root', default='data')
    parser.add_argument('--time_window', type=int, default=3, help='How many nearby (past or future) frames to see, length=2*time_window+1')
    parser.add_argument('--pose_norm', type=int, default=1, choices=[0, 1], help='Normalize pose or not')
    parser.add_argument('--data_mode', type=str, default='kp', choices=['op', 'kp'], help='Choose which part of data to use')
    # Model
    parser.add_argument('--num_blocks', type=int, default=2, help='How many residual blocks to use')
    parser.add_argument('--num_features', type=int, default=512, help='Number of channels in the intermediate layers')
    # Optimization
    parser.add_argument('--bs', type=int, default=512, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='How many workers for data loading')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--decay_gamma', type=float, default=0.9)
    # Logging
    parser.add_argument('--out_dir', default='outputs', help='Path to save models and logs')
    parser.add_argument('--out_name', default='temp', help='An unique name for each training')
    parser.add_argument('--use_tfboard', default=1, choices=[0, 1], help='Whether or not to use TensorBoard')
    parser.add_argument('--log_steps', default=100, help='How many steps to record once')
    parser.add_argument('--ckpt_steps', default=0, help='How many steps to save ckpt once. 0 means saving ckpt each epoch')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Called with args:')
    print(args)

    ## Dataset
    dataset_path = os.path.join(args.root, 'Human3.6M')
    dataset_train1 = DatasetHuman36m(root=dataset_path, mode='train', time_window=args.time_window, 
        pose_norm=bin_to_bool(args.pose_norm), output_mode=args.data_mode)
    dataset_path2 = os.path.join(args.root, 'MADS')
    dataset_train2 = DatasetMads(root=dataset_path2, mode='train', time_window=args.time_window, 
        pose_norm=bin_to_bool(args.pose_norm), output_mode=args.data_mode)
    dataset_list = [dataset_train1, dataset_train2]
    dataset_train = DatasetConcat(dataset_list)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs, shuffle=True, 
        num_workers=args.num_workers, drop_last=True)

    dataset_val = DatasetHuman36m(root=dataset_path, mode='val', time_window=args.time_window, 
        pose_norm=bin_to_bool(args.pose_norm), output_mode=args.data_mode)
    
    steps_per_epoch = len(dataset_train)//args.bs

    ## Model
    assert args.time_window > 0
    # Multiple frame
    if args.data_mode == 'op':
        in_channels = len(REST_INDS)*3
    elif args.data_mode == 'kp':
        in_channels = len(REST_INDS)*2
    else:
        raise NotImplementedError

    model = BasicTemporalModel(in_channels=in_channels, num_features=args.num_features, num_blocks=args.num_blocks, time_window=args.time_window)

    model.cuda()
    model = nn.DataParallel(model)

    ## Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    ## Logging
    output_dir = os.path.join(args.out_dir, args.out_name)    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
        f.write(str(args))

    if bin_to_bool(args.use_tfboard):
        tblogger = SummaryWriter(output_dir)
    else:
        tblogger = None

    # Average in each log_steps
    avg_loss = 0.
    stat_dict = OrderedDict()
    for key in ['All', 'LToe', 'LHeel', 'RToe', 'RHeel']:
        stat_dict[key] = {'Accuracy': 0., 'TP': 0., 'TN': 0., 'FP': 0., 'FN': 0.}

    ## Training
    model.train()

    global_step = 0
    best_f1 = 0.    # To pick the best model on val set
    best_pre = 0.
    best_rec = 0.
    best_epoch = -1
    for epoch in range(args.num_epochs):
        scheduler.step()
        dataiterator_train = iter(dataloader_train)

        for step in range(steps_per_epoch):
            inputs = next(dataiterator_train)
            # NOTE: Need to check if there exists any valid label
            mask = inputs[-1]
            if mask.sum() == 0:
                continue

            global_step += 1
            optimizer.zero_grad()
            
            body, label, mask = inputs
            body = body.cuda()

            label = label.cuda()
            mask = mask.cuda()
            pred_logit = model(body)
            pred_prob = torch.sigmoid(pred_logit)
            loss = balanced_loss(pred_logit, label, mask)
            loss.backward()
            optimizer.step()

            # Tracking stats
            avg_loss += loss.item()/args.log_steps
            pred_label = pred_prob.detach()
            pred_label[pred_label<0.5] = 0
            pred_label[pred_label>=0.5] = 1

            mask_np = mask.cpu().numpy()
            # Recover it back to {-1, 0, 1}
            label_np = label.cpu().numpy()+(mask_np-1)
            pred_label_np = pred_label.cpu().numpy()
            res = label_np==pred_label_np
            # NOTE: Original definition
            TN = np.logical_and(pred_label_np==0, label_np==0)
            TP = np.logical_and(pred_label_np==1, label_np==1)
            FN = np.logical_and(pred_label_np==0, label_np==1)
            FP = np.logical_and(pred_label_np==1, label_np==0)

            for i, key in enumerate(stat_dict):
                if key == 'All':
                    stat_dict[key]['Accuracy'] += res.sum()/mask_np.sum()/args.log_steps
                    stat_dict[key]['TP'] += TP.sum()
                    stat_dict[key]['TN'] += TN.sum()
                    stat_dict[key]['FP'] += FP.sum()
                    stat_dict[key]['FN'] += FN.sum()
                else:
                    stat_dict[key]['Accuracy'] += res[:, i-1].sum()/mask_np[:, i-1].sum()/args.log_steps
                    stat_dict[key]['TP'] += TP[:, i-1].sum()
                    stat_dict[key]['TN'] += TN[:, i-1].sum()
                    stat_dict[key]['FP'] += FP[:, i-1].sum()
                    stat_dict[key]['FN'] += FN[:, i-1].sum()

            # Logging
            if global_step % args.log_steps == 0:
                log_str = 'Global Step: {}, Train Epoch: {}/{} [{}/{}], Loss: {:.6f}\n'.format(global_step, epoch+1, args.num_epochs, step+1, steps_per_epoch, avg_loss)

                if bin_to_bool(args.use_tfboard):
                    tblogger.add_scalar('loss', avg_loss, global_step)

                for key in stat_dict:
                    acc = stat_dict[key]['Accuracy']
                    tp = stat_dict[key]['TP']
                    tn = stat_dict[key]['TN']
                    fp = stat_dict[key]['FP']
                    fn = stat_dict[key]['FN']

                    del stat_dict[key]['TP']
                    del stat_dict[key]['TN']
                    del stat_dict[key]['FP']
                    del stat_dict[key]['FN']

                    eps = 1e-6
                    pre = tp/(tp+fp+eps)
                    rec = tp/(tp+fn+eps)
                    f1 = 2.*pre*rec/(pre+rec+eps)

                    stat_dict[key]['Precision'] = pre
                    stat_dict[key]['Recall'] = rec
                    stat_dict[key]['F1'] = f1

                    log_str += '\t\t{}: Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:4f}\n'.format(key, acc, pre, rec, f1)
               

                    if bin_to_bool(args.use_tfboard):
                        for subkey in stat_dict[key]:
                            name = 'train/'+key+'/'+subkey
                            tblogger.add_scalar(name, stat_dict[key][subkey], global_step)
                print(log_str)

                avg_loss = 0.
                for key in stat_dict:
                    stat_dict[key]['Accuracy'] = 0.
                    stat_dict[key]['TP'] = 0.
                    stat_dict[key]['TN'] = 0.
                    stat_dict[key]['FP'] = 0.
                    stat_dict[key]['FN'] = 0.

            # Save ckpt
            if args.ckpt_steps > 0 and global_step % args.ckpt_steps == 0:
                curr_f1, curr_pre, curr_rec = evaluate(model, dataset_val, args, global_step, tblogger)
                if curr_pre > best_pre:
                    best_pre = curr_pre
                    best = True
                else:
                    best = False
                print('Saving ckpt...')
                save_ckpt(output_dir, args, epoch, global_step, model, optimizer, best)
                print('Ckpt is saved!')

        # Save ckpt after each epoch
        if args.ckpt_steps == 0:
            curr_f1, curr_pre, curr_rec = evaluate(model, dataset_val, args, global_step, tblogger)
            if curr_pre > best_pre:
                best_pre = curr_pre
                best = True
                best_epoch = epoch
            else:
                best = False
            print('Saving ckpt...')
            save_ckpt(output_dir, args, epoch, global_step, model, optimizer, best)
            print('Ckpt is saved!')

    print('Finish training, best precision: {} at epoch {}'.format(best_pre, best_epoch))


if __name__ == '__main__':
    main()
