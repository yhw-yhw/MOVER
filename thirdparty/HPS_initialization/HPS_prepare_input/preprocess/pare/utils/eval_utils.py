"""
Parts of the code are adapted from https://github.com/akanazawa/hmr
"""
import os
import json
import yaml
import cv2
import torch
import numpy as np
from loguru import logger

SMPL_OR_JOINTS = np.array([0, 1, 2, 4, 5, 16, 17, 18, 19])


def joint_angle_error(pred_mat, gt_mat):
    """
    Compute the geodesic distance between the two input matrices.
    :param pred_mat: predicted rotation matrices. Shape: ( Seq, 24, 3, 3)
    :param gt_mat: ground truth rotation matrices. Shape: ( Seq, 24, 3, 3)
    :return: Mean geodesic distance between input matrices.
    """

    gt_mat = gt_mat[:, SMPL_OR_JOINTS, :, :]
    pred_mat = pred_mat[:, SMPL_OR_JOINTS, :, :]

    # Reshape the matrices into B x 3 x 3 arrays
    r1 = np.reshape(pred_mat, [-1, 3, 3])
    r2 = np.reshape(gt_mat, [-1, 3, 3])

    # Transpose gt matrices
    r2t = np.transpose(r2, [0, 2, 1])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(r1, r2t)

    angles = []
    # Convert rotation matrix to axis angle representation and find the angle
    for i in range(r1.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))

    return np.mean(np.array(angles))


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)

    re_per_joint = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1))
    re = re_per_joint.mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re, re_per_joint


def find_best_ckpt(cfg_file, use_mpjpe=False, new_version=False):
    cfg = yaml.load(open(cfg_file))
    log_dir = '/'.join(cfg_file.split('/')[:-1])
    val_results_log = os.path.join(log_dir, 'val_accuracy_results_3dpw.json')
    acc_results = json.load(open(val_results_log))

    check_freq = int(cfg['TRAINING']['CHECK_VAL_EVERY_N_EPOCH'])

    acc_arr = []
    for acc in acc_results:
        if acc[1]['val_mpjpe'] < 10: continue
        acc_arr.append([acc[1]['val_mpjpe'], acc[1]['val_pampjpe']])

    accuracy = np.array(acc_arr)

    if new_version:
        epochs_list = [int(acc[2]) for acc in acc_results if acc[1]['val_mpjpe'] >= 10]
        best_mpjpe_epoch = epochs_list[accuracy[:, 0].argmin()]
        best_pampjpe_epoch = epochs_list[accuracy[:, 1].argmin()]
    else:
        best_mpjpe_epoch = (accuracy[:,0].argmin() + 1) * check_freq - 1
        best_pampjpe_epoch = (accuracy[:, 1].argmin() + 1) * check_freq - 1

    best_epoch = best_mpjpe_epoch if use_mpjpe else best_pampjpe_epoch

    ckpt_file = None

    for root, dirs, files in os.walk(log_dir, topdown=False):
        for f in files:
            if f.endswith('.ckpt'):
                if int(f.split('=')[-1].split('.')[0]) == best_epoch:
                    ckpt_file = os.path.join(root, f)

    assert ckpt_file is not None, 'Best performing checkpoint file could not be found'

    logger.info(f'Found best performing checkpoint: \"{ckpt_file}\"')
    logger.info(f'Performance MPJPE: {accuracy[accuracy[:,1].argmin(), 0]:.2f}, '
                f'PA-MPJPE: {accuracy[accuracy[:,1].argmin(), 1]:.2f}')

    return ckpt_file


