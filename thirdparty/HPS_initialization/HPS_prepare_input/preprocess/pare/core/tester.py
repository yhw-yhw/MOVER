import os
import cv2
import time
import torch
import joblib
import colorsys
import numpy as np
from tqdm import tqdm
from loguru import logger
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from ..models import PARE, HMR
from .config import update_hparams
from ..utils.vibe_renderer import Renderer
from ..utils.pose_tracker import run_posetracker
from ..utils.train_utils import load_pretrained_model
from ..dataset.inference import Inference, ImageFolder
from ..utils.smooth_pose import smooth_pose
from ..utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    convert_crop_coords_to_orig_img,
    prepare_rendering_results,
)


MIN_NUM_FRAMES = 0


class PARETester:
    def __init__(self, args):
        self.args = args
        self.model_cfg = update_hparams(args.cfg)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self._build_model()
        self._load_pretrained_model()
        self.model.eval()

    def _build_model(self):
        # ========= Define PARE model ========= #
        model_cfg = self.model_cfg

        if model_cfg.METHOD == 'pare':
            model = PARE(
                backbone=model_cfg.PARE.BACKBONE,
                num_joints=model_cfg.PARE.NUM_JOINTS,
                softmax_temp=model_cfg.PARE.SOFTMAX_TEMP,
                num_features_smpl=model_cfg.PARE.NUM_FEATURES_SMPL,
                focal_length=model_cfg.DATASET.FOCAL_LENGTH,
                img_res=model_cfg.DATASET.IMG_RES,
                pretrained=model_cfg.TRAINING.PRETRAINED,
                iterative_regression=model_cfg.PARE.ITERATIVE_REGRESSION,
                num_iterations=model_cfg.PARE.NUM_ITERATIONS,
                iter_residual=model_cfg.PARE.ITER_RESIDUAL,
                shape_input_type=model_cfg.PARE.SHAPE_INPUT_TYPE,
                pose_input_type=model_cfg.PARE.POSE_INPUT_TYPE,
                pose_mlp_num_layers=model_cfg.PARE.POSE_MLP_NUM_LAYERS,
                shape_mlp_num_layers=model_cfg.PARE.SHAPE_MLP_NUM_LAYERS,
                pose_mlp_hidden_size=model_cfg.PARE.POSE_MLP_HIDDEN_SIZE,
                shape_mlp_hidden_size=model_cfg.PARE.SHAPE_MLP_HIDDEN_SIZE,
                use_keypoint_features_for_smpl_regression=model_cfg.PARE.USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION,
                use_heatmaps=model_cfg.DATASET.USE_HEATMAPS,
                use_keypoint_attention=model_cfg.PARE.USE_KEYPOINT_ATTENTION,
                use_postconv_keypoint_attention=model_cfg.PARE.USE_POSTCONV_KEYPOINT_ATTENTION,
                use_scale_keypoint_attention=model_cfg.PARE.USE_SCALE_KEYPOINT_ATTENTION,
                keypoint_attention_act=model_cfg.PARE.KEYPOINT_ATTENTION_ACT,
                use_final_nonlocal=model_cfg.PARE.USE_FINAL_NONLOCAL,
                use_branch_nonlocal=model_cfg.PARE.USE_BRANCH_NONLOCAL,
                use_hmr_regression=model_cfg.PARE.USE_HMR_REGRESSION,
                use_coattention=model_cfg.PARE.USE_COATTENTION,
                num_coattention_iter=model_cfg.PARE.NUM_COATTENTION_ITER,
                coattention_conv=model_cfg.PARE.COATTENTION_CONV,
                use_upsampling=model_cfg.PARE.USE_UPSAMPLING,
                deconv_conv_kernel_size=model_cfg.PARE.DECONV_CONV_KERNEL_SIZE,
                use_soft_attention=model_cfg.PARE.USE_SOFT_ATTENTION,
                num_branch_iteration=model_cfg.PARE.NUM_BRANCH_ITERATION,
                branch_deeper=model_cfg.PARE.BRANCH_DEEPER,
                num_deconv_layers=model_cfg.PARE.NUM_DECONV_LAYERS,
                num_deconv_filters=model_cfg.PARE.NUM_DECONV_FILTERS,
                use_resnet_conv_hrnet=model_cfg.PARE.USE_RESNET_CONV_HRNET,
                use_position_encodings=model_cfg.PARE.USE_POS_ENC,
                use_mean_camshape=model_cfg.PARE.USE_MEAN_CAMSHAPE,
                use_mean_pose=model_cfg.PARE.USE_MEAN_POSE,
                init_xavier=model_cfg.PARE.INIT_XAVIER,
            ).to(self.device)
        elif model_cfg.METHOD == 'spin':
            model = HMR(
                backbone=model_cfg.SPIN.BACKBONE,
                img_res=model_cfg.DATASET.IMG_RES,
                pretrained=model_cfg.TRAINING.PRETRAINED,
                p=model_cfg.TRAINING.DROPOUT_P,
                estimate_var=model_cfg.SPIN.ESTIMATE_UNCERTAINTY,
            ).to(self.device)
        else:
            logger.error(f'{model_cfg.METHOD} is undefined!')
            exit()

        return model

    def _load_pretrained_model(self):
        # ========= Load pretrained weights ========= #
        if self.args.ckpt == 'spin':
            logger.warning('CKPT file is not provided, using SPIN weights')
        else:
            logger.info(f'Loading pretrained model from {self.args.ckpt}')
            ckpt = torch.load(self.args.ckpt)['state_dict']
            load_pretrained_model(self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
            logger.info(f'Loaded pretrained weights from \"{self.args.ckpt}\"')

    def run_tracking(self, video_file, image_folder):
        # ========= Run tracking ========= #
        if self.args.tracking_method == 'pose':
            if not os.path.isabs(video_file):
                video_file = os.path.join(os.getcwd(), video_file)
            tracking_results = run_posetracker(video_file, staf_folder=self.args.staf_dir, display=self.args.display)
        else:
            # run multi object tracker
            mot = MPT(
                device=self.device,
                batch_size=self.args.tracker_batch_size,
                display=self.args.display,
                detector_type=self.args.detector,
                output_format='dict',
                yolo_img_size=self.args.yolo_img_size,
            )
            tracking_results = mot(image_folder)

        # remove tracklets if num_frames is less than MIN_NUM_FRAMES
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                del tracking_results[person_id]

        return tracking_results

    def run_detector(self, image_folder):
        pass

    @torch.no_grad()
    def run_on_image_folder(self, image_folder, detection_results, bbox_scale=1.0):
        dataset = ImageFolder(image_folder, bboxes, joints2d, scale=bbox_scale)

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=8)

        pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

        for batch in dataloader:
            if has_keypoints:
                batch, nj2d = batch
                norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

            batch = batch.to(self.device)
            batch_size = batch.shape[0]
            output = self.model(batch)

            pred_cam.append(output['pred_cam'])  # [:, :, :3].reshape(batch_size, -1))
            pred_verts.append(output['smpl_vertices'])  # .reshape(batch_size * seqlen, -1, 3))
            pred_pose.append(output['pred_pose'])  # [:,:,3:75].reshape(batch_size * seqlen, -1))
            pred_betas.append(output['pred_shape'])  # [:, :,75:].reshape(batch_size * seqlen, -1))
            pred_joints3d.append(output['smpl_joints3d'])  # .reshape(batch_size * seqlen, -1, 3))

        pred_cam = torch.cat(pred_cam, dim=0)
        pred_verts = torch.cat(pred_verts, dim=0)
        pred_pose = torch.cat(pred_pose, dim=0)
        pred_betas = torch.cat(pred_betas, dim=0)
        pred_joints3d = torch.cat(pred_joints3d, dim=0)

        del batch

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        pare_results = output_dict

        return pare_results

    @torch.no_grad()
    def run_on_video(self, tracking_results, image_folder, orig_width, orig_height, bbox_scale=1.0):
        # ========= Run PARE on each person ========= #
        logger.info(f'Running PARE on each tracklet...')

        pare_results = {}
        for person_id in tqdm(list(tracking_results.keys())):
            bboxes = joints2d = None

            if self.args.tracking_method == 'bbox':
                bboxes = tracking_results[person_id]['bbox']
            elif self.args.tracking_method == 'pose':
                joints2d = tracking_results[person_id]['joints2d']

            frames = tracking_results[person_id]['frames']

            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=bbox_scale,
            )

            bboxes = dataset.bboxes
            frames = dataset.frames
            has_keypoints = True if joints2d is not None else False

            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=8)

            pred_cam, pred_verts, pred_pose, pred_betas, \
            pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.to(self.device)
                batch_size = batch.shape[0]
                output = self.model(batch)

                pred_cam.append(output['pred_cam'])  # [:, :, :3].reshape(batch_size, -1))
                pred_verts.append(output['smpl_vertices'])  # .reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['pred_pose'])  # [:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['pred_shape'])  # [:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['smpl_joints3d'])  # .reshape(batch_size * seqlen, -1, 3))
                smpl_joints2d.append(output['smpl_joints2d'])

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)

            del batch

            # ========= [Optional] run Temporal SMPLify to refine the results ========= #
            # if args.run_smplify and args.tracking_method == 'pose':
            #     norm_joints2d = np.concatenate(norm_joints2d, axis=0)
            #     norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
            #     norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)
            #
            #     # Run Temporal SMPLify
            #     update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
            #     new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
            #         pred_rotmat=pred_pose,
            #         pred_betas=pred_betas,
            #         pred_cam=pred_cam,
            #         j2d=norm_joints2d,
            #         device=device,
            #         batch_size=norm_joints2d.shape[0],
            #         pose2aa=False,
            #     )
            #
            #     # update the parameters after refinement
            #     print(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
            #     pred_verts = pred_verts.cpu()
            #     pred_cam = pred_cam.cpu()
            #     pred_pose = pred_pose.cpu()
            #     pred_betas = pred_betas.cpu()
            #     pred_joints3d = pred_joints3d.cpu()
            #     pred_verts[update] = new_opt_vertices[update]
            #     pred_cam[update] = new_opt_cam[update]
            #     pred_pose[update] = new_opt_pose[update]
            #     pred_betas[update] = new_opt_betas[update]
            #     pred_joints3d[update] = new_opt_joints3d[update]
            #
            # elif args.run_smplify and args.tracking_method == 'bbox':
            #     print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
            #     print('[WARNING] Continuing without running Temporal SMPLify!..')

            # ========= Save results to a pickle file ========= #
            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            smpl_joints2d = smpl_joints2d.cpu().numpy()

            if self.args.smooth:
                min_cutoff = self.args.min_cutoff  # 0.004
                beta = self.args.beta  # 1.5
                logger.info(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
                pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                                   min_cutoff=min_cutoff, beta=beta)

            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=orig_width,
                img_height=orig_height
            )
            logger.info('Converting smpl keypoints 2d to original image coordinate')

            smpl_joints2d = convert_crop_coords_to_orig_img(
                bbox=bboxes,
                keypoints=smpl_joints2d,
                crop_size=self.model_cfg.DATASET.IMG_RES,
            )

            output_dict = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': joints2d,
                'smpl_joints2d': smpl_joints2d,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

            pare_results[person_id] = output_dict
        return pare_results

    def render_results(self, pare_results, image_folder, output_img_folder, output_path,
                       orig_width, orig_height, num_frames):
        # ========= Render results as a single video ========= #
        renderer = Renderer(
            resolution=(orig_width, orig_height),
            orig_img=True,
            wireframe=self.args.wireframe
        )

        logger.info(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(pare_results, num_frames)
        if self.args.exp in ['pare', 'eft', 'spin']:
            mesh_color = joblib.load(f'data/demo_mesh_colors_{self.args.exp}.npy')
        else:
            mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in pare_results.keys()}

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            if self.args.sideview:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']
                frame_kp = person_data['joints2d']

                mc = mesh_color[person_id]

                mesh_filename = None

                if self.args.save_obj:
                    mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )

                if self.args.draw_keypoints:
                    for idx, pt in enumerate(frame_kp):
                        cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0,255,0), -1)

                if self.args.sideview:
                    side_img = renderer.render(
                        side_img,
                        frame_verts,
                        cam=frame_cam,
                        color=mc,
                        angle=270,
                        axis=[0, 1, 0],
                    )

            if self.args.sideview:
                img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if self.args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.args.display:
            cv2.destroyAllWindows()