import json
import os
from collections import OrderedDict

import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def save_ckpt(output_dir, args, epoch, step, model, optimizer, best=False):
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if isinstance(model, nn.DataParallel):
        model = model.module
    save_dict = {
        'epoch': epoch,
        'step': step,
        'batch_size': args.bs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    # torch.save(save_dict, save_name)
    # Save another replica
    if best:
        save_name = os.path.join(ckpt_dir, 'model_best.pth')
        torch.save(save_dict, save_name)


def evaluate(model, dataset, args, global_step, tblogger):
    model.eval()

    print('\nEvaluating on validation set: ......')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, drop_last=False)
    dataiterator = iter(dataloader)
    steps_per_epoch = len(dataset)//args.bs

    stat_dict = OrderedDict()
    for key in ['All', 'LToe', 'LHeel', 'RToe', 'RHeel']:
        stat_dict[key] = {'Accuracy': 0., 'TP': 0., 'TN': 0., 'FP': 0., 'FN': 0., 'Count': 0.}

    for step in range(steps_per_epoch+1):
        inputs = next(dataiterator)
        # if bin_to_bool(args.out_foot):
        #     body, foot, label, mask = inputs
        #     body, foot = body.cuda(), foot.cuda()
        #     pred_prob = model(body, foot)
        # else:
        #     body, label, mask = inputs
        #     body = body.cuda()
        #     pred_prob = model(body)

        body, label, mask = inputs
        body = body.cuda()
        pred_prob = model(body)

        pred_label = pred_prob.detach()
        pred_label[pred_label<0.5] = 0
        pred_label[pred_label>=0.5] = 1

        pred_label_np = pred_label.cpu().numpy()

        mask_np = mask.numpy()
        # Recover it back to {-1, 0, 1}
        label_np = label.cpu().numpy()+(mask_np-1)
        res = label_np==pred_label_np
        # # NOTE: We have much more positive examples, so we need to swap the label for these metrics
        # TN = np.logical_and(pred_label_np==1, label_np==1)
        # TP = np.logical_and(pred_label_np==0, label_np==0)
        # FN = np.logical_and(pred_label_np==1, label_np==0)
        # FP = np.logical_and(pred_label_np==0, label_np==1)
        # NOTE: Original definition
        TN = np.logical_and(pred_label_np==0, label_np==0)
        TP = np.logical_and(pred_label_np==1, label_np==1)
        FN = np.logical_and(pred_label_np==0, label_np==1)
        FP = np.logical_and(pred_label_np==1, label_np==0)

        for i, key in enumerate(stat_dict):
            if key == 'All':
                stat_dict[key]['Accuracy'] += res.sum()
                stat_dict[key]['Count'] += mask_np.sum()
                stat_dict[key]['TP'] += TP.sum()
                stat_dict[key]['TN'] += TN.sum()
                stat_dict[key]['FP'] += FP.sum()
                stat_dict[key]['FN'] += FN.sum()
            else:
                stat_dict[key]['Accuracy'] += res[:, i-1].sum()
                stat_dict[key]['Count'] += mask_np[:, i-1].sum()
                stat_dict[key]['TP'] += TP[:, i-1].sum()
                stat_dict[key]['TN'] += TN[:, i-1].sum()
                stat_dict[key]['FP'] += FP[:, i-1].sum()
                stat_dict[key]['FN'] += FN[:, i-1].sum()

    # Summary
    log_str = 'Results on validation set:\n'
    eps = 1e-6
    for key in stat_dict:
        acc = stat_dict[key]['Accuracy']/stat_dict[key]['Count']
        tp = stat_dict[key]['TP']
        tn = stat_dict[key]['TN']
        fp = stat_dict[key]['FP']
        fn = stat_dict[key]['FN']

        stat_dict[key]['Accuracy'] = acc
        del stat_dict[key]['Count']
        del stat_dict[key]['TP']
        del stat_dict[key]['TN']
        del stat_dict[key]['FP']
        del stat_dict[key]['FN']

        pre = tp/(tp+fp+eps)
        rec = tp/(tp+fn+eps)
        f1 = 2.*pre*rec/(pre+rec+eps)

        stat_dict[key]['Precision'] = pre
        stat_dict[key]['Recall'] = rec
        stat_dict[key]['F1'] = f1

        if key == 'All':
            ret1 = f1
            ret2 = pre
            ret3 = rec
        log_str += '\t\t{}: Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:4f}\n'.format(key, acc, pre, rec, f1)

        if bin_to_bool(args.use_tfboard):
            for subkey in stat_dict[key]:
                name = 'val/'+key+'/'+subkey
                tblogger.add_scalar(name, stat_dict[key][subkey], global_step)

    print(log_str)
    model.train()
    return ret1, ret2, ret3


def bin_to_bool(var):
    if var == 0:
        return False
    else:
        return True


def balanced_loss(logit, label, mask):
    # # Magic numbers from training set stats (noisy_ground)
    # magics = [0.15363571, 0.13683611, 0.20319784, 0.16715136]
    # Magic numbers from training set stats (noisy_ground_2 and noisy_ground_2_mads)
    # Pos: [41025. 37771. 41159. 37981.]
    # Neg: [24848. 28786. 24788. 28500.]
    # Invalid: [56815. 56131. 56741. 56207.]
    # NOTE: Newest data stat
    # Pos: [39624. 38000. 42514. 37708.]
    # Neg: [24626. 28000. 24996. 29272.]
    # Invalid: [58370. 56620. 55110. 55640.]
    magics = [0.621492 , 0.7368421, 0.5879475, 0.7762809]
    loss = 0
    for i, magic in enumerate(magics):
        pos_weight=torch.Tensor([magic]).cuda()
        temp_loss = F.binary_cross_entropy_with_logits(logit[:,i], label[:,i], pos_weight=pos_weight, reduction='none')*mask[:, i]
        loss += temp_loss.sum()/mask[:, i].sum()
    return loss


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    # Pick the most confident detection
    scores = [np.mean(kp[kp[:, 2] > -1, 2]) for kp in kps]
    try:
        kp = kps[np.argmax(scores)]
    except:
        kp = np.zeros((25, 3), dtype=np.float32)
    return kp