import torch
import numpy as np

from ..models.head.smpl_head import SMPL
from ..core.config import SMPL_MODEL_DIR
from .one_euro_filter import OneEuroFilter


def smooth_pose(pred_pose, pred_betas, min_cutoff=0.004, beta=0.7):
    # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
    # beta: Increasing the speed coefficient(beta) decreases speed lag.

    one_euro_filter = OneEuroFilter(
        np.zeros_like(pred_pose[0]),
        pred_pose[0],
        min_cutoff=min_cutoff,
        beta=beta,
    )

    smpl = SMPL(model_path=SMPL_MODEL_DIR)

    pred_pose_hat = np.zeros_like(pred_pose)

    # initialize
    pred_pose_hat[0] = pred_pose[0]

    pred_verts_hat = []
    pred_joints3d_hat = []

    smpl_output = smpl(
        betas=torch.from_numpy(pred_betas[0]).unsqueeze(0),
        body_pose=torch.from_numpy(pred_pose[0, 1:]).unsqueeze(0),
        global_orient=torch.from_numpy(pred_pose[0, 0:1]).unsqueeze(0),
        pose2rot=False,
    )
    pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
    pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())

    for idx, pose in enumerate(pred_pose[1:]):
        idx += 1

        t = np.ones_like(pose) * idx
        pose = one_euro_filter(t, pose)
        pred_pose_hat[idx] = pose

        smpl_output = smpl(
            betas=torch.from_numpy(pred_betas[idx]).unsqueeze(0),
            body_pose=torch.from_numpy(pred_pose_hat[idx, 1:]).unsqueeze(0),
            global_orient=torch.from_numpy(pred_pose_hat[idx, 0:1]).unsqueeze(0),
            pose2rot=False,
        )
        pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
        pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())

    return np.vstack(pred_verts_hat), pred_pose_hat, np.vstack(pred_joints3d_hat)