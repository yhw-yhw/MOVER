import torch
import torch.nn as nn

from ..losses.losses import projected_keypoint_loss


class EFTLoss(nn.Module):
    def __init__(
            self,
            keypoint_loss_weight=5.,
            beta_loss_weight=0.001,
            openpose_train_weight=0.,
            gt_train_weight=1.,
            leg_orientation_loss_weight=0.005,
            loss_weight=60.,
    ):
        super(EFTLoss, self).__init__()
        self.criterion_keypoints = nn.MSELoss(reduction='none')

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.beta_loss_weight = beta_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.openpose_train_weight = openpose_train_weight
        self.leg_orientation_loss_weight = leg_orientation_loss_weight

    def forward(self, pred, gt):
        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_projected_keypoints_2d = pred['smpl_joints2d']

        gt_keypoints_2d = gt['keypoints']

        loss_leg_orientation, gt_keypoints_2d = get_loss_leg_orientation(gt_keypoints_2d, pred_projected_keypoints_2d)

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            criterion=self.criterion_keypoints,
        )

        loss_keypoints *= self.keypoint_loss_weight
        # loss_keypoints_3d *= self.keypoint_loss_weight
        loss_regr_betas_no_reject = torch.mean(pred_betas ** 2) * self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()
        loss_leg_orientation *= self.leg_orientation_loss_weight

        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            # 'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_cam': loss_cam,
            'loss/loss_leg_orientation': loss_leg_orientation,
            'loss/loss_regr_betas_no_reject': loss_regr_betas_no_reject,
        }

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        loss_dict['loss/total_loss'] = loss

        return loss, loss_dict


def normalize_2dvector(gt_bone_orientation):
    gt_bone_orientation_norm = torch.norm(gt_bone_orientation, dim=1)  # (N)
    # gt_boneOri_leftLeg[:,0] = gt_boneOri_leftLeg[:,0]/gt_boneOri_leftLeg_norm
    # gt_boneOri_leftLeg[:,1] = gt_boneOri_leftLeg[:,1]/gt_boneOri_leftLeg_norm
    gt_bone_orientation = gt_bone_orientation / gt_bone_orientation_norm

    return gt_bone_orientation, gt_bone_orientation_norm


def get_loss_leg_orientation(gt_keypoints_2d, pred_keypoints_2d):
    # Ignore hips and hip centers, foot
    LENGTH_THRESHOLD = 0.0089  # 1/112.0     #at least it should be 5 pixel

    device = gt_keypoints_2d.device

    # Disable Hips by default

    gt_keypoints_2d[:, 2 + 25, 2] = 0
    gt_keypoints_2d[:, 3 + 25, 2] = 0
    gt_keypoints_2d[:, 14 + 25, 2] = 0

    # #Compute angle knee to ankle orientation
    gt_boneOri_leftLeg = gt_keypoints_2d[:, 5 + 25, :2] - gt_keypoints_2d[:, 4 + 25, :2] # Left lower leg orientation #(N,2)
    gt_boneOri_leftLeg, leftLegLeng = normalize_2dvector(gt_boneOri_leftLeg)

    if leftLegLeng > LENGTH_THRESHOLD:
        leftLegValidity = gt_keypoints_2d[:, 5 + 25, 2] * gt_keypoints_2d[:, 4 + 25, 2]
        pred_boneOri_leftLeg = pred_keypoints_2d[:, 5 + 25, :2] - pred_keypoints_2d[:, 4 + 25, :2]
        pred_boneOri_leftLeg, _ = normalize_2dvector(pred_boneOri_leftLeg)
        loss_legOri_left = torch.ones(1).to(device) - torch.dot(gt_boneOri_leftLeg.view(-1),
                                                                     pred_boneOri_leftLeg.view(-1))
    else:
        loss_legOri_left = torch.zeros(1).to(device)
        leftLegValidity = torch.zeros(1).to(device)

    gt_boneOri_rightLeg = gt_keypoints_2d[:, 0 + 25, :2] - gt_keypoints_2d[:, 1 + 25,
                                                           :2]  # Right lower leg orientation
    gt_boneOri_rightLeg, rightLegLeng = normalize_2dvector(gt_boneOri_rightLeg)
    if rightLegLeng > LENGTH_THRESHOLD:

        rightLegValidity = gt_keypoints_2d[:, 0 + 25, 2] * gt_keypoints_2d[:, 1 + 25, 2]
        pred_boneOri_rightLeg = pred_keypoints_2d[:, 0 + 25, :2] - pred_keypoints_2d[:, 1 + 25, :2]
        pred_boneOri_rightLeg, _ = normalize_2dvector(pred_boneOri_rightLeg)
        loss_legOri_right = torch.ones(1).to(device) - torch.dot(gt_boneOri_rightLeg.view(-1),
                                                                      pred_boneOri_rightLeg.view(-1))
    else:
        loss_legOri_right = torch.zeros(1).to(device)
        rightLegValidity = torch.zeros(1).to(device)
    # print("leftLegLeng: {}, rightLegLeng{}".format(leftLegLeng,rightLegLeng ))
    loss_legOri = leftLegValidity * loss_legOri_left + rightLegValidity * loss_legOri_right

    # loss_legOri = torch.zeros(1).to(self.device)
    # if leftLegValidity.item():
    #     loss_legOri = loss_legOri + (pred_boneOri_leftLeg).mean()
    # if rightLegValidity.item():
    #     loss_legOri = loss_legOri + self.criterion_regr(gt_boneOri_rightLeg, pred_boneOri_rightLeg)
    # print(loss_legOri)

    # Disable Foots
    gt_keypoints_2d[:, 5 + 25, 2] = 0  # Left foot
    gt_keypoints_2d[:, 0 + 25, 2] = 0  # Right foot

    return loss_legOri, gt_keypoints_2d