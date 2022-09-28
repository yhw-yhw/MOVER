import os
import cv2
import sys
import time
import copy
import torch
import joblib
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .loss import EFTLoss
from ..models import HMR, SMPL
from ..utils.renderer import Renderer
from ..dataset import BaseDataset, LMDBDataset
from ..utils.image_utils import denormalize_images
from ..utils.eval_utils import reconstruction_error
from ..utils.train_utils import load_pretrained_model
from ..core.config import SMPL_MODEL_DIR, JOINT_REGRESSOR_H36M
from ..utils.geometry import convert_weak_perspective_to_perspective
from ..core.constants import H36M_TO_J14, H36M_TO_J17, J24_TO_J14, J24_TO_J17


class EFTFitter():
    def __init__(self, hparams):
        super(EFTFitter, self).__init__()

        self.hparams = hparams
        self.log = self.hparams.LOG
        self.device = 'cuda' if torch.cuda.is_available() else sys.exit('CUDA is not available')


        logdir = f'{self.hparams.DATASET.VAL_DS}/{time.strftime("%d-%m-%Y_%H-%M-%S")}_' \
                 f'{self.hparams.EXP_NAME}_' \
                 f'iter-{self.hparams.MAX_EXEMPLAR_ITER}' \
                 f'lr-{self.hparams.EXEMPLAR_LR}'

        self.hparams.LOG_DIR = os.path.join(self.hparams.LOG_DIR, logdir)
        self.hparams.SAVE_IMAGES = self.hparams.SAVE_IMAGES if self.log else False

        if self.log:
            logger.add(
                os.path.join(self.hparams.LOG_DIR, 'finetune.log'),
                level='INFO',
                colorize=False,
            )

        logger.info(f'Hyperparameters: \n {hparams}')

        self.model = HMR(
            backbone=self.hparams.SPIN.BACKBONE,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
        ).to(self.device)

        self.optimizer = self.configure_optimizers()

        self.loss_fn = EFTLoss(
            keypoint_loss_weight=self.hparams.LOSS.KEYPOINT_LOSS_WEIGHT,
            beta_loss_weight=self.hparams.LOSS.BETA_LOSS_WEIGHT,
            openpose_train_weight=self.hparams.LOSS.OPENPOSE_TRAIN_WEIGHT,
            gt_train_weight=self.hparams.LOSS.GT_TRAIN_WEIGHT,
            leg_orientation_loss_weight=self.hparams.LOSS.LEG_ORIENTATION_LOSS_WEIGHT,
            loss_weight=self.hparams.LOSS.LOSS_WEIGHT,
        )

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False
        ).to(self.device)

        render_resolution = self.hparams.DATASET.RENDER_RES
        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=render_resolution,
            faces=self.smpl.faces,
        )

        # self.finetune_ds = self.finetune_dataset()

        logger.info(">>> Load Pretrained model: {}".format(self.hparams.PRETRAINED_CKPT))
        ckpt = torch.load(self.hparams.PRETRAINED_CKPT)['state_dict']
        load_pretrained_model(self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
        # opt_ckpt = torch.load(self.hparams.PRETRAINED_CKPT)['optimizer_states'][0]
        # self.optimizer.load_state_dict(opt_ckpt)
        self.backup_model()


    def init_evaluation_variables(self):
        dataset_length = len(self.finetune_ds)

        self.joint_mapper_h36m = H36M_TO_J17 if self.finetune_ds.dataset == 'mpi-inf-3dhp' else H36M_TO_J14
        self.joint_mapper_gt = J24_TO_J17 if self.finetune_ds.dataset == 'mpi-inf-3dhp' else J24_TO_J14

        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float().to(self.device)

        # stores mean mpjpe/pa-mpjpe values for all validation dataset samples
        self.init_val_mpjpe = np.zeros(dataset_length)
        self.init_val_pampjpe = np.zeros(dataset_length)
        self.ft_val_mpjpe = np.zeros(dataset_length)
        self.ft_val_pampjpe = np.zeros(dataset_length)

        # Store SMPL parameters
        self.init_smpl_pose = np.zeros((dataset_length, 24, 3, 3))
        self.init_smpl_betas = np.zeros((dataset_length, 10))
        self.init_smpl_camera = np.zeros((dataset_length, 3))
        self.init_pred_joints = np.zeros((dataset_length, len(self.joint_mapper_h36m), 3))

        self.ft_smpl_pose = np.zeros((dataset_length, 24, 3, 3))
        self.ft_smpl_betas = np.zeros((dataset_length, 10))
        self.ft_smpl_camera = np.zeros((dataset_length, 3))
        self.ft_pred_joints = np.zeros((dataset_length, len(self.joint_mapper_h36m), 3))

        # This dict is used to store metrics and metadata for a more detailed analysis
        # per-joint, per-sequence, occluded-sequences etc.
        self.evaluation_results = {
            'imgname': [],
            'dataset_name': [],
            'init_mpjpe': np.zeros((len(self.finetune_ds), 14)),
            'init_pampjpe': np.zeros((len(self.finetune_ds), 14)),
            'ft_mpjpe': np.zeros((len(self.finetune_ds), 14)),
            'ft_pampjpe': np.zeros((len(self.finetune_ds), 14)),
        }


    def finetune(self):
        self.finetune_ds = self.finetune_dataset()
        self.init_evaluation_variables()

        for step, batch in enumerate(tqdm(self.finetune_dataloader())):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            ft_pred = self.finetune_step(batch, step)

            if self.hparams.EVALUATE:
                self.eval_one_batch(batch, ft_pred, step)

            if self.hparams.SAVE_IMAGES:
                self.visualize_results(batch, ft_pred, step)

        # Report and save results
        logger.info(f'Init MPJPE: {1000 * self.init_val_mpjpe.mean()}')
        logger.info(f'Init PA-MPJPE: {1000 * self.init_val_pampjpe.mean()}')
        logger.info(f'EFT MPJPE: {1000 * self.ft_val_mpjpe.mean()}')
        logger.info(f'EFT PA-MPJPE: {1000 * self.ft_val_pampjpe.mean()}')

        if self.log:
            joblib.dump(
                self.evaluation_results,
                os.path.join(self.hparams.LOG_DIR, f'evaluation_results_{self.hparams.DATASET.VAL_DS}.pkl')
            )

            np.savez(
                os.path.join(self.hparams.LOG_DIR, f'model_preds_{self.hparams.DATASET.VAL_DS}.npz'),
                init_pred_joints=self.init_pred_joints,
                init_pose=self.init_smpl_pose,
                init_betas=self.init_smpl_betas,
                init_camera=self.init_smpl_camera,
                ft_pred_joints=self.ft_pred_joints,
                ft_pose=self.ft_smpl_pose,
                ft_betas=self.ft_smpl_betas,
                ft_camera=self.ft_smpl_camera,
            )

    def finetune_step(self, batch, batch_nb):
        self.reload_model()

        ft_pred = {}
        output_backup = {}
        for eft_iter in range(self.hparams.MAX_EXEMPLAR_ITER):
            self.model.train()
            self.exemplar_training_mode()
            images = batch['img']
            ft_pred = self.model(images)

            if eft_iter == 0:
                output_backup = {f'init_{k}': v.clone() for k,v in ft_pred.items()}
            # pred:
            # smpl_vertices - torch.Size([1, 6890, 3])
            # smpl_joints3d - torch.Size([1, 49, 3])
            # smpl_joints2d - torch.Size([1, 49, 2])
            # pred_cam_t - torch.Size([1, 3])
            # pred_segm_mask - torch.Size([1, 25, 56, 56])
            # pred_pose - torch.Size([1, 24, 3, 3])
            # pred_cam - torch.Size([1, 3])
            # pred_shape - torch.Size([1, 10])

            loss, loss_dict = self.loss_fn(pred=ft_pred, gt=batch)
            # print(eft_iter, loss.item(), end='\r')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.hparams.MIN_LOSS > 0:
                if loss.item() < self.hparams.MIN_LOSS and eft_iter >= self.hparams.MIN_EXEMPLAR_ITER:
                    # print('final loss', loss.item(), eft_iter, batch_nb)
                    break


        ft_pred.update(output_backup)
        return ft_pred

    def eval_one_batch(self, batch, ft_pred, batch_nb):
        curr_batch_size = batch['img'].shape[0]
        imgnames = batch['imgname']
        dataset_names = batch['dataset_name']

        J_regressor_batch = self.J_regressor[None, :].expand(curr_batch_size, -1, -1)

        # joint mapper is problematic because of the way ConcatDataset works
        # joint_mapper_h36m = constants.H36M_TO_J17 if batch['dataset_name'][0] == 'mpi-inf-3dhp' \
        #     else constants.H36M_TO_J14

        # gt_keypoints_3d = batch['pose_3d']
        # For 3DPW get the 14 common joints from the rendered shape
        if self.finetune_ds.dataset in ['3dpw', '3dpw-all', '3doh']:
            # for 3dpw and 3doh datasets obtain the gt 3d joints
            # from pose and shape parameters
            gt_pose, gt_betas = batch['pose'].to(self.device), batch['betas'].to(self.device)
            gt_vertices = self.smpl(
                global_orient=gt_pose[:, :3],
                body_pose=gt_pose[:, 3:],
                betas=gt_betas
            ).vertices
            gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, self.joint_mapper_h36m, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
        else:
            gt_keypoints_3d = batch['pose_3d'].to(self.device)
            gt_keypoints_3d = gt_keypoints_3d[:, self.joint_mapper_gt, :-1]

        ft_vertices = ft_pred['smpl_vertices']
        # Get 14 predicted joints from the mesh
        ft_keypoints_3d = torch.matmul(J_regressor_batch, ft_vertices)
        ft_pelvis = ft_keypoints_3d[:, [0], :].clone()
        ft_keypoints_3d = ft_keypoints_3d[:, self.joint_mapper_h36m, :]
        ft_keypoints_3d = ft_keypoints_3d - ft_pelvis
        # Absolute error (MPJPE)
        ft_error = torch.sqrt(((ft_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).detach().cpu().numpy()
        idx_start = batch_nb * self.hparams.DATASET.BATCH_SIZE
        idx_stop = batch_nb * self.hparams.DATASET.BATCH_SIZE + curr_batch_size
        # Reconstuction_error
        ft_r_error, ft_r_error_per_joint = reconstruction_error(
            ft_keypoints_3d.detach().cpu().numpy(),
            gt_keypoints_3d.detach().cpu().numpy(),
            reduction=None,
        )
        ft_error_per_joint = torch.sqrt(((ft_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).detach().cpu().numpy()

        init_vertices = ft_pred['init_smpl_vertices']
        # Get 14 predicted joints from the mesh
        init_keypoints_3d = torch.matmul(J_regressor_batch, init_vertices)
        init_pelvis = init_keypoints_3d[:, [0], :].clone()
        init_keypoints_3d = init_keypoints_3d[:, self.joint_mapper_h36m, :]
        init_keypoints_3d = init_keypoints_3d - init_pelvis
        # Absolute error (MPJPE)
        init_error = torch.sqrt(((init_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).detach().cpu().numpy()

        # Reconstuction_error
        init_r_error, init_r_error_per_joint = reconstruction_error(
            init_keypoints_3d.detach().cpu().numpy(),
            gt_keypoints_3d.detach().cpu().numpy(),
            reduction=None,
        )
        init_error_per_joint = torch.sqrt(((init_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).detach().cpu().numpy()

        # Save Results
        self.init_smpl_pose[idx_start:idx_stop, :] = ft_pred['init_pred_pose'].detach().cpu().numpy()
        self.init_smpl_betas[idx_start:idx_stop, :] = ft_pred['init_pred_shape'].detach().cpu().numpy()
        self.init_smpl_camera[idx_start:idx_stop, :] = ft_pred['init_pred_cam'].detach().cpu().numpy()
        self.init_pred_joints[idx_start:idx_stop, :] = init_keypoints_3d.detach().cpu().numpy()

        self.ft_smpl_pose[idx_start:idx_stop, :] = ft_pred['pred_pose'].detach().cpu().numpy()
        self.ft_smpl_betas[idx_start:idx_stop, :] = ft_pred['pred_shape'].detach().cpu().numpy()
        self.ft_smpl_camera[idx_start:idx_stop, :] = ft_pred['pred_cam'].detach().cpu().numpy()
        self.ft_pred_joints[idx_start:idx_stop, :] = ft_keypoints_3d.detach().cpu().numpy()

        self.ft_val_mpjpe[idx_start:idx_stop] = ft_error
        self.ft_val_pampjpe[idx_start:idx_stop] = ft_r_error
        self.init_val_mpjpe[idx_start:idx_stop] = init_error
        self.init_val_pampjpe[idx_start:idx_stop] = init_r_error

        self.evaluation_results['ft_mpjpe'][idx_start:idx_stop] = ft_error_per_joint[:, :14]
        self.evaluation_results['ft_pampjpe'][idx_start:idx_stop] = ft_r_error_per_joint[:, :14]
        self.evaluation_results['init_mpjpe'][idx_start:idx_stop] = init_error_per_joint[:, :14]
        self.evaluation_results['init_pampjpe'][idx_start:idx_stop] = init_r_error_per_joint[:, :14]
        self.evaluation_results['imgname'] += imgnames
        self.evaluation_results['dataset_name'] += dataset_names

    def visualize_results(self, batch, ft_pred, batch_idx, has_error_metrics=True):
        images = batch['disp_img']
        images = denormalize_images(images)
        curr_batch_size = images.shape[0]

        idx_start = batch_idx * self.hparams.DATASET.BATCH_SIZE
        idx_stop = batch_idx * self.hparams.DATASET.BATCH_SIZE + curr_batch_size

        init_vertices = ft_pred['init_smpl_vertices'].detach()
        ft_vertices = ft_pred['smpl_vertices'].detach()

        if has_error_metrics:
            ft_error = self.ft_val_mpjpe[idx_start:idx_stop]
            ft_r_error = self.ft_val_pampjpe[idx_start:idx_stop]
            ft_per_joint_error = self.evaluation_results['ft_mpjpe'][idx_start:idx_stop]

            init_error = self.init_val_mpjpe[idx_start:idx_stop]
            init_r_error = self.init_val_pampjpe[idx_start:idx_stop]
            init_per_joint_error = self.evaluation_results['init_mpjpe'][idx_start:idx_stop]

        ########### convert camera parameters to display image params ###########
        ft_cam = ft_pred['pred_cam'].detach()
        ft_cam_t = convert_weak_perspective_to_perspective(
            ft_cam,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.RENDER_RES,
        )

        init_cam = ft_pred['init_pred_cam'].detach()
        init_cam_t = convert_weak_perspective_to_perspective(
            init_cam,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.RENDER_RES,
        )

        images_ft = self.renderer.visualize_tb(
            ft_vertices,
            ft_cam_t,
            images,
            nb_max_img=1,
            sideview=True,
            joint_labels=ft_per_joint_error * 1000. if has_error_metrics else None,
            kp_2d=batch['keypoints'],
            skeleton_type='spin',
        )

        images_init = self.renderer.visualize_tb(
            init_vertices,
            init_cam_t,
            images,
            nb_max_img=1,
            sideview=True,
            joint_labels=init_per_joint_error * 1000. if has_error_metrics else None,
            kp_2d=batch['keypoints'],
            skeleton_type='spin',
        )

        # self.logger.experiment.add_image('pred_shape', images_pred, self.global_step)
        images_ft = images_ft.cpu().numpy().transpose(1, 2, 0) * 255
        images_ft = np.clip(images_ft, 0, 255).astype(np.uint8)

        images_init = images_init.cpu().numpy().transpose(1, 2, 0) * 255
        images_init = np.clip(images_init, 0, 255).astype(np.uint8)

        if has_error_metrics:
            # draw the errors as text on saved images
            images_ft = cv2.putText(
                images_ft, f'e: {ft_error[0] * 1000:.1f}, re: {ft_r_error[0] * 1000:.1f}',
                (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)
            )
            images_init = cv2.putText(
                images_init, f'e: {init_error[0] * 1000:.1f}, re: {init_r_error[0] * 1000:.1f}',
                (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)
            )

        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(save_dir, f'{batch_idx:05d}_init.jpg'),
            cv2.cvtColor(images_init, cv2.COLOR_BGR2RGB)
        )
        cv2.imwrite(
            os.path.join(save_dir, f'{batch_idx:05d}_ft.jpg'),
            cv2.cvtColor(images_ft, cv2.COLOR_BGR2RGB)
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.hparams.EXEMPLAR_LR,
            weight_decay=0,
        )

    def finetune_dataset(self):
        return eval(f'{self.hparams.DATASET.LOAD_TYPE}Dataset')(
            options=self.hparams.DATASET,
            dataset=self.hparams.DATASET.VAL_DS,
            is_train=False,
            num_images=self.hparams.DATASET.NUM_IMAGES,
        )

    def finetune_dataloader(self):
        return DataLoader(
            dataset=self.finetune_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            shuffle=False,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
        )

    def backup_model(self):
        logger.info(">>> Model status saved!")
        self.model_backup = copy.deepcopy(self.model.state_dict())
        self.optimizer_backup = copy.deepcopy(self.optimizer.state_dict())

    def reload_model(self):
        # logger.info(">>> Model status has been reloaded to initial!")
        self.model.load_state_dict(self.model_backup)
        self.optimizer.load_state_dict(self.optimizer_backup)

    def exemplar_training_mode(self):
        for module in self.model.modules():
            if not type(module):
                continue

            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                for m in module.parameters():
                    m.requires_grad = False

            if isinstance(module, nn.Dropout):
                module.eval()
                for m in module.parameters():
                    m.requires_grad = False