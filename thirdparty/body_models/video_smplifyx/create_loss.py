

from ..smplifyx import fitting as single_view_fitting
from .fitting_video_loss import *
from .fitting_video_orientation_temporal_loss import SMPLifyBodyOrientLoss, MultiViewSMPLifyBodyOrientLoss

def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return single_view_fitting.SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return single_view_fitting.SMPLifyCameraInitLoss(**kwargs)
    elif loss_type == 'body_orient':
        return SMPLifyBodyOrientLoss(**kwargs)
    elif loss_type == 'body_orient_multiview':
        return MultiViewSMPLifyBodyOrientLoss(**kwargs)
    elif loss_type == 'multiview_smplify':
        return MultiViewSMPLifyLoss(**kwargs)
    elif loss_type == '3D_joint_loss': # it could direct work.
        return SMPLifyLoss3D(**kwargs)
    elif loss_type == '3D_joint_hands_loss': # it could direct work.
        return SMPLifyLoss3D_withHands(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))
        