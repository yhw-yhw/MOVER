import torch

from ..models import SMPL
from .losses import camera_fitting_loss, body_fitting_loss
from ..core import config, constants
from ..utils.geometry import convert_weak_perspective_to_perspective, batch_rodrigues

# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from .prior import MaxMixturePrior

class SMPLify():
    """Implementation of single-stage SMPLify.""" 
    def __init__(
            self,
            cam_step_size=1e-2,
            pose_step_size=1e-2,
            batch_size=66,
            pose_num_iters=100,
            cam_num_iters=100,
            focal_length=5000,
            img_res=224,
            device=torch.device('cuda'),
            use_weak_perspective=False,
            use_all_joints_for_camera=False,
            camera_opt_params=('global_orient', 'camera_translation'), # 'focal_length', 'camera_rotation'
            optimize_cam_only=False,
    ):

        # Store options
        self.device = device
        self.img_res = img_res
        self.cam_step_size = cam_step_size
        self.pose_step_size = pose_step_size
        self.focal_length = focal_length
        self.optimize_cam_only = optimize_cam_only
        self.camera_opt_params = camera_opt_params
        self.use_weak_perspective = use_weak_perspective
        self.use_all_joints_for_camera = use_all_joints_for_camera

        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.pose_num_iters = pose_num_iters
        self.cam_num_iters = cam_num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder='data',
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # Load SMPL model
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(self.device)

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = init_pose.shape[0]

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad=False
        betas.requires_grad=False
        global_orient.requires_grad=True
        camera_translation.requires_grad = True
        focal_length = self.focal_length
        # camera_rotation = torch.eye(3, device=self.device,
        #                             dtype=camera_translation.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        # camera_rotation = camera_rotation.clone()

        camera_rotation_aa = torch.zeros(3, device=camera_translation.device).unsqueeze(0).expand(batch_size, -1)
        camera_rotation_aa = camera_rotation_aa.clone()

        camera_opt_params = []

        for opt_param in self.camera_opt_params:
            if opt_param == 'camera_translation':
                camera_opt_params.append(camera_translation)
            if opt_param == 'focal_length':
                focal_length = torch.tensor([self.focal_length], device=self.device,
                                            dtype=camera_translation.dtype).repeat(batch_size)
                focal_length.requires_grad = True
                camera_opt_params.append(focal_length)
            if opt_param == 'global_orient':
                camera_opt_params.append(global_orient)
            if opt_param == 'camera_rotation':
                camera_rotation_aa.requires_grad = True
                camera_opt_params.append(camera_rotation_aa)


        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.cam_step_size, betas=(0.9, 0.999))

        for i in range(self.cam_num_iters):
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas)
            model_joints = smpl_output.joints

            camera_rotation = batch_rodrigues(camera_rotation_aa)

            loss = camera_fitting_loss(
                model_joints,
                camera_translation,
                init_cam_t,
                camera_center,
                joints_2d,
                joints_conf,
                camera_rotation=camera_rotation,
                focal_length=focal_length,
                use_weak_perspective=self.use_weak_perspective,
                img_res=self.img_res,
                use_all_joints_for_camera=self.use_all_joints_for_camera
            )

            print(f'{i:04d}/{self.cam_num_iters} Camera loss: {loss.item():.3f}', end='\r')
            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # Fix camera translation after optimizing camera
        if 'camera_translation' in self.camera_opt_params:
            camera_translation.requires_grad = False
        if 'focal_length' in self.camera_opt_params:
            focal_length.requires_grad = False
        if 'camera_rotation' in self.camera_opt_params:
            camera_rotation_aa.requires_grad = False
            camera_rotation = batch_rodrigues(camera_rotation_aa)

        if self.optimize_cam_only:
            pass
        else:
            # Step 2: Optimize body joints
            # Optimize only the body pose and global orientation of the body
            body_pose.requires_grad = True
            betas.requires_grad = True
            global_orient.requires_grad = True
            camera_translation.requires_grad = False
            body_opt_params = [body_pose, betas, global_orient]

            # For joints ignored during fitting, set the confidence to 0
            joints_conf[:, self.ign_joints] = 0.

            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.pose_step_size, betas=(0.9, 0.999))
            for i in range(self.pose_num_iters):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints
                loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior,
                                         focal_length=focal_length, camera_rotation=camera_rotation,
                                         use_weak_perspective=self.use_weak_perspective, img_res=self.img_res)
                print(f'{i:04d}/{self.pose_num_iters} Pose loss: {loss.item():.3f}', end='\r')

                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(
                body_pose,
                betas,
                model_joints,
                camera_translation,
                camera_center,
                joints_2d,
                joints_conf,
                self.pose_prior,
                camera_rotation=camera_rotation,
                focal_length=focal_length,
                output='reprojection'
            )

        if self.use_weak_perspective:
            camera_translation = convert_weak_perspective_to_perspective(
                camera_translation,
                focal_length=focal_length,
                img_res=self.img_res,
            )

        # camera_rotation = batch_rodrigues(camera_rotation)

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation, reprojection_loss, focal_length, camera_rotation

    def get_fitting_loss(self, pose, betas, cam_t, camera_center, keypoints_2d):
        """Given body and camera parameters, compute reprojection loss value.
        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = pose.shape[0]

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, cam_t, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        return reprojection_loss
