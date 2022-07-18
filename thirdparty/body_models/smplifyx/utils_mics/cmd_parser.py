# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import configargparse
import ast 

def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'PyTorch implementation of SMPLifyX'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='SMPLifyX')

    parser.add_argument('--single',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Print info messages during the process')

    parser.add_argument('--data_folder',
                        default=os.getcwd(),
                        help='The directory that contains the data.')
    parser.add_argument('--max_persons', type=int, default=3,
                        help='The maximum number of persons to process')
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--loss_type', default='smplify', type=str,
                        help='The type of loss to use')
    parser.add_argument('--interactive',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Print info messages during the process')
    parser.add_argument('--save_meshes',
                        type=lambda arg: arg.lower() == 'true',
                        default=True,
                        help='Save final output meshes')
    parser.add_argument('--visualize',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Display plots while running the optimization')
    parser.add_argument('--degrees', type=float, default=[0, 90, 180, 270],
                        help='Degrees of rotation for rendering the final' +
                        ' result')
    parser.add_argument('--use_cuda',
                        type=lambda arg: arg.lower() == 'true',
                        default=True,
                        help='Use CUDA for the computations')
    parser.add_argument('--dataset', default='hands_cmu_gt', type=str,
                        help='The name of the dataset that will be used')
    parser.add_argument('--joints_to_ign', default=-1, type=int,
                        nargs='*',
                        help='Indices of joints to be ignored')
    parser.add_argument('--output_folder',
                        default='output',
                        type=str,
                        help='The folder where the output is stored')
    parser.add_argument('--img_folder', type=str, default='images',
                        help='The folder where the images are stored')
    parser.add_argument('--keyp_folder', type=str, default='keypoints',
                        help='The folder where the keypoints are stored')
    parser.add_argument('--summary_folder', type=str, default='summaries',
                        help='Where to store the TensorBoard summaries')
    parser.add_argument('--result_folder', type=str, default='results',
                        help='The folder with the pkls of the output' +
                        ' parameters')
    parser.add_argument('--mesh_folder', type=str, default='meshes',
                        help='The folder where the output meshes are stored')
    parser.add_argument('--gender_lbl_type', default='none',
                        choices=['none', 'gt', 'pd'], type=str,
                        help='The type of gender label to use')
    parser.add_argument('--gender', type=str,
                        default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL' +
                        'model')
    parser.add_argument('--float_dtype', type=str, default='float32',
                        help='The types of floats used')
    parser.add_argument('--model_type', default='smpl', type=str,
                        choices=['smpl', 'smplh', 'smplx'],
                        help='The type of the model that we will fit to the' +
                        ' data.')
    parser.add_argument('--camera_type', type=str, default='user',
                        choices=['persp','user'],
                        help='The type of camera used')
    parser.add_argument('--calib_path', type=str, default='calibration',
                        help='The folder where calibration files (xml) are stored')
    parser.add_argument('--calib_path_oriJ3d', type=str, default='None',
                        help='The folder where calibration files (xml) are stored')
    parser.add_argument('--optim_jaw', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the jaw pose')
    parser.add_argument('--optim_hands', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the hand pose')
    parser.add_argument('--optim_expression', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the expression')
    parser.add_argument('--optim_shape', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the shape space')

    parser.add_argument('--model_folder',
                        default='models',
                        type=str,
                        help='The directory where the models are stored.')
    parser.add_argument('--use_joints_conf', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the confidence scores for the optimization')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The size of the batch')
    parser.add_argument('--num_gaussians',
                        default=8,
                        type=int,
                        help='The number of gaussian for the Pose Mixture' +
                        ' Prior.')
    parser.add_argument('--use_pca', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the low dimensional PCA space for the hands')
    parser.add_argument('--num_pca_comps', default=6, type=int,
                        help='The number of PCA components for the hand.')
    parser.add_argument('--flat_hand_mean', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use the flat hand as the mean pose')
    parser.add_argument('--body_prior_type', default='mog', type=str,
                        help='The type of prior that will be used to' +
                        ' regularize the optimization. Can be a Mixture of' +
                        ' Gaussians (mog)')
    parser.add_argument('--left_hand_prior_type', default='mog', type=str,
                        choices=['mog', 'l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' left hand. Can be a Mixture of' +
                        ' Gaussians (mog)')
    parser.add_argument('--right_hand_prior_type', default='mog', type=str,
                        choices=['mog', 'l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' right hand. Can be a Mixture of' +
                        ' Gaussians (mog)')
    parser.add_argument('--jaw_prior_type', default='l2', type=str,
                        choices=['l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' jaw.')
    parser.add_argument('--use_vposer', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use the VAE pose embedding')
    parser.add_argument('--vposer_ckpt', type=str, default='',
                        help='The path to the V-Poser checkpoint')
    # Left/Right shoulder and hips
    parser.add_argument('--init_joints_idxs', nargs='*', type=int,
                        default=[9, 12, 2, 5],
                        help='Which joints to use for initializing the camera')
    parser.add_argument('--body_tri_idxs', default='5.12,2.9',
                        type=lambda x: [list(map(int, pair.split('.')))
                                        for pair in x.split(',')],
                        help='The indices of the joints used to estimate' +
                        ' the initial depth of the camera. The format' +
                        ' should be vIdx1.vIdx2,vIdx3.vIdx4')

    # ! this is important: to define the template formate.
    parser.add_argument("--random_sample_order", type=int, default=-1, help="random sample the start frame along a video.")
    parser.add_argument("--all_frame", type=int, default=-1, help="random sample the start frame along a video.")
    parser.add_argument("--start_frame", type=int, default=-1, help="random sample the start frame along a video.")
    
    # ! noise input
    parser.add_argument('--noise_kind', type=int, default=-1, help='')
    parser.add_argument('--noise_value', type=float, default=0.0, help='')
    
    parser.add_argument('--prior_folder', type=str, default='prior',
                        help='The folder where the prior is stored')
    parser.add_argument('--focal_length',
                        default=5000,
                        type=float,
                        help='Value of focal length.')
    parser.add_argument('--rho',
                        default=100,
                        type=float,
                        help='Value of constant of robust loss')
    parser.add_argument('--interpenetration',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to use the interpenetration term')
    parser.add_argument('--penalize_outside',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Penalize outside')
    parser.add_argument('--data_weights', nargs='*',
                        default=[1, ] * 5, type=float,
                        help='The weight of the data term')
    parser.add_argument('--body_pose_prior_weights',
                        default=[4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
                        nargs='*',
                        type=float,
                        help='The weights of the body pose regularizer')
    parser.add_argument('--shape_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float, nargs='*',
                        help='The weights of the Shape regularizer')
    parser.add_argument('--expr_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float, nargs='*',
                        help='The weights of the Expressions regularizer')
    parser.add_argument('--face_joints_weights',
                        default=[0.0, 0.0, 0.0, 2.0], type=float,
                        nargs='*',
                        help='The weights for the facial keypoints' +
                        ' for each stage of the optimization')
    parser.add_argument('--hand_joints_weights',
                        default=[0.0, 0.0, 0.0, 2.0],
                        type=float, nargs='*',
                        help='The weights for the 2D joint error of the hands')
    parser.add_argument('--jaw_pose_prior_weights',
                        nargs='*',
                        help='The weights of the pose regularizer of the' +
                        ' hands')
    parser.add_argument('--hand_pose_prior_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float, nargs='*',
                        help='The weights of the pose regularizer of the' +
                        ' hands')

    parser.add_argument('--coll_loss_weights',
                        default=[0.0, 0.0, 0.0, 2.0], type=float,
                        nargs='*',
                        help='The weight for the collision term')

    parser.add_argument('--depth_loss_weight', default=1e2, type=float,
                        help='The weight for the regularizer for the' +
                        ' z coordinate of the camera translation')
    parser.add_argument('--df_cone_height', default=0.5, type=float,
                        help='The default value for the height of the cone' +
                        ' that is used to calculate the penetration distance' +
                        ' field')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--point2plane', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use point to plane distance')
    parser.add_argument('--part_segm_fn', default='', type=str,
                        help='The file with the part segmentation for the' +
                        ' faces of the model')
    parser.add_argument('--ign_part_pairs', default=None,
                        nargs='*', type=str,
                        help='Pairs of parts whose collisions will be ignored')
    parser.add_argument('--use_hands', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the hand keypoints in the SMPL' +
                        'optimization process')
    parser.add_argument('--use_face', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the facial keypoints in the optimization' +
                        ' process')
    parser.add_argument('--use_face_contour', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the dynamic contours of the face')
    parser.add_argument('--side_view_thsh',
                        default=25,
                        type=float,
                        help='This is thresholding value that determines' +
                        ' whether the human is captured in a side view.' +
                        'If the pixel distance between the shoulders is less' +
                        ' than this value, two initializations of SMPL fits' +
                        ' are tried.')
    parser.add_argument('--optim_type', type=str, default='adam',
                        help='The optimizer used')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='The learning rate for the algorithm')
    parser.add_argument('--gtol', type=float, default=1e-8,
                        help='The tolerance threshold for the gradient')
    parser.add_argument('--ftol', type=float, default=2e-9,
                        help='The tolerance threshold for the function')
    parser.add_argument('--maxiters', type=int, default=100,
                        help='The maximum iterations for the optimization')
    parser.add_argument('--sigma', default=50, type=float,
                        help='The tolerance in pixels for view weight adjustment')
    parser.add_argument('--start_opt_stage', default=0, type=int,
                        help='Which smplify-x optimization stage to start with: [0:5]')
    parser.add_argument('--end_opt_stage', default=0, type=int,
                        help='Which smplify-x optimization stage to start with: [0:5]')
    parser.add_argument('--beta_precomputed', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use the pre-computed beta')
    parser.add_argument('--beta_path', type=str, default='',
                        help='The file where precomputed shape are stored')

    # demo_joint argument
    parser.add_argument('--img_path', type=str, default=None, help='')
    parser.add_argument('--img_dir', type=str, default=None, help='')
    parser.add_argument('--img_dir_det', type=str, default=None, help='')
    parser.add_argument('--width', type=int, default=640, help='')
    parser.add_argument('--height', type=int, default=360, help='')
    parser.add_argument('--pare_dir', type=str, default=None, help='')
    parser.add_argument('--st2_render_body_dir', type=str, default=None, help='used in Body filter to see filtered body results.')
    parser.add_argument('--posa_dir', type=str, default=None, help='')
    parser.add_argument('--scene_result_dir', type=str, default=None, help='')
    parser.add_argument('--scene_result_path', type=str, default=None, help='')
    parser.add_argument('--scanned_path', type=str, default=None, help='')
    
    parser.add_argument('--cam_inc_fn', type=str, default=None, help='')
    parser.add_argument('--scene_visualize',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Display plots while running the optimization')

    parser.add_argument('--save_dir', type=str, default=None, help='')
    parser.add_argument('--save', help='True or False flag, input should be either "True" or "False".',
        type=ast.literal_eval, default=True)
    parser.add_argument('--debug', help='True or False flag, input should be either "True" or "False".',
        type=ast.literal_eval, default=True)
    parser.add_argument('--pre_load', help='True or False flag, input should be either "True" or "False".',
        type=ast.literal_eval, default=True)
    parser.add_argument('--pre_load_pare_pose', help='True or False flag, input should be either "True" or "False".',
        type=ast.literal_eval, default=True)
    
    # ! pure_scene_loss flag 
    parser.add_argument('--pure_scene_loss', help='True or False flag, input should be either "True" or "False".',
        type=ast.literal_eval, default=False)
    parser.add_argument('--update_orientation_specific', help='True or False flag, input should be either "True" or "False".',
        type=ast.literal_eval, default=False)
    parser.add_argument('--update_gp_camera', help='True or False flag, input should be either "True" or "False".',
        type=ast.literal_eval, default=False)

    parser.add_argument('--visualization', help='True or False flag, input should be either "True" or "False".',
        type=ast.literal_eval, default=True)
    # used for tensorboard visualization
    # parser.add_argument('--tb_debug', help='True or False flag, input should be either "True" or "False".',
    #     type=ast.literal_eval, default=True)
    parser.add_argument('--idx', type=int, default=-1, help='')

    # filter_img_idx: is used for single image.
    parser.add_argument('--filter_img_idx', type=int, default=-1, help='')

    parser.add_argument('--img_list',
                        default='1_10_100',
                        type=lambda x: [int(pair)
                                        for pair in x.split('_')],
                        help='image idx list')
    parser.add_argument('--start_stage', type=int, default=1, help='')
    parser.add_argument('--end_stage', type=int, default=4, help='')
    
    # split video into a consequtive way.
    parser.add_argument('--split_video', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--resample_in_sdf', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    
    parser.add_argument('--use_pose_estimate_camera', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')

    parser.add_argument('--RECALCULATE_HCI_INFO', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--no_render', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')

    parser.add_argument('--REINIT_ORIENT_BY_BODY', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    
    parser.add_argument('--REINIT_SCALE_POSITION_BY_BODY', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')

    parser.add_argument('--DEGENERATE_DEPTH', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--higherChair', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--RENEW_OPT', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')

    parser.add_argument('--degenerate_scale', default=0.1, type=float,
                        help='')

    # ! flag for reinit orientation.
    parser.add_argument('--use_total3d_reinit', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
                        

    # process_id
    parser.add_argument('--process_id', type=int, default=0, help='')
    parser.add_argument('--stage3_idx', default=3, type=int,
                        help='Which smplify-x optimization stage to start with: [0:5]')
    parser.add_argument('--scene_init_model', type=str, default=None, help='')
    parser.add_argument('--load_all_scene', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    
    parser.add_argument('--ground_contact_path', type=str, default=None, help='')
    # ground plane loss
    parser.add_argument('--ground_plane_support', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--gp_support_weights_init', default=0.0, type=float,
                        help='')
    parser.add_argument('--gp_support_weights',
                        default=[0.0, 0.0, 0.0, 0.0, 0.0],
                        nargs='*', type=float,
                        help='The weights of the ground plane support loss')
    # ground contact loss
    parser.add_argument('--ground_contact_support', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--gp_contact_weights',
                        default=[0.0, 0.0, 0.0, 0.0, 0.0],
                        nargs='*', type=float,
                        help='The weights of the ground plane support loss')
    
    parser.add_argument('--body_segments_dir', default="",
                        type=str,
                        help='')

    # # sdf penetration
    parser.add_argument('--sdf_penetration', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--sdf_penetration_loss_weight', default=[0.0, 0.0, 0.0, 0.0, 0.0], nargs='*', type=float,
                        help='')
    parser.add_argument('--update_scene', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--update_body', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    # ## contact
    parser.add_argument('--contact',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--rho_contact',
                        default=1,
                        type=float,
                        help='Value of constant of robust loss')
    parser.add_argument('--contact_angle',
                        default=45,
                        type=float,
                        help='used to refine normals. (angle in degrees)')
    parser.add_argument('--contact_loss_weights',
                        default=[0.0, 0.0, 0.0, 0.0, 0.0], type=float,
                        nargs='*',
                        help='The weight for the contact term')
    parser.add_argument('--contact_body_parts',
                        default=['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs'], type=str,
                        nargs='*',
                        help='')
    
    # contact assign label
    parser.add_argument('--body2obj_npy_path', type=str, default=None, help='')
    parser.add_argument('--preload',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')

    parser.add_argument('--scene',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--scene_loss_weight',
                        default=[0.0, 0.0, 0.0, 0.0, 0.0], type=float,
                        nargs='*',
                        help='The weight for the 3D scene term')
    parser.add_argument('--pare_pose_prior',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--pare_pose_weight',
                        default=[0.0, 0.0, 0.0, 0.0, 0.0], type=float,
                        nargs='*',
                        help='The weight for the pare pose prior term')
    parser.add_argument('--load_scalenet_cam',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--cams_params', default='0.14,2.9',
                        type=lambda x: [float(pair)
                                        for pair in x.split(',')],
                        help='The indices of the joints used to estimate' +
                        ' the initial depth of the camera. The format' +
                        ' should be vIdx1.vIdx2,vIdx3.vIdx4')
    parser.add_argument('--cams_scalenet_fn',
                        default=None, type=str,
                        help='path to scalenet camera estimation')

    parser.add_argument('--ordinal_depth',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--ordinal_depth_loss_weight',
                        default=[0.0, 0.0, 0.0, 0.0, 0.0], type=float,
                        nargs='*',
                        help='The weight for the 3D scene term')
                        
    parser.add_argument('--inner_robust_sdf',
                        type=float,
                        default=None,
                        help='')

    parser.add_argument('--inner_robust_depth',
                        type=float,
                        default=None,
                        help='')

    ## use for video process and define loss
    parser.add_argument('--video_smooth',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--motion_smooth_prior',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--constant_velocity',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--loss_use_sum',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--motion_prior_weight',
                        default=0.0, type=float,
                        help='The weight for the 3D scene term')
    parser.add_argument('--smooth_2d_weight',
                        default=0.0, type=float,
                        help='The weight for the 3D scene term')
                        
    parser.add_argument('--smooth_3d_weight',
                        default=0.0, type=float,
                        help='The weight for the 3D scene term')

    parser.add_argument('--use_body2scene_conf',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--use_video',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--cluster',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--stage3_kind_flag', type=int, default=31, help='')
    
    # * add input fps
    parser.add_argument('--input_fps', type=int, default=30, help='')
    
    parser.add_argument('--use_human_depth',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--depth_robustifier',
                        default=5,
                        type=float,
                        help='Value of constant of depth_robustifier loss')

    ### codebase selection
    parser.add_argument('--use_scene_loss',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')

    ### * chose the last one model as the best one
    
    parser.add_argument('--use_final_last_model',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    # used for scene-assisted
    # parser.add_argument('--st4_video',type=lambda arg: arg.lower() in ['true', '1'],
    #                     default=True,
    #                     help='')
    parser.add_argument('--stage4_split_idx',
                        default=0, type=int,
                        help='stage 4 split idx.')
    parser.add_argument('--human_batch',
                        default=1, type=int,
                        help='human_batch.')

    parser.add_argument('--orientation_sample_num',
                        default=6, type=int,
                        help='orientation sampled num.')

    # use them for scene initialization
    parser.add_argument('--USE_MASK',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--UPDATE_CAMERA_EXTRIN',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--USE_CAD_SIZE',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--USE_ONE_DOF_SCALE',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--UPDATE_OBJ_SCALE',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--USE_INIT_SCENE',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')

    parser.add_argument('--running_opt',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=True,
                        help='')
    parser.add_argument('--input_obj_idx',
                        default=-1, type=int,
                        help='human_batch.')

    parser.add_argument('--SIGMOID_FOR_SCALE',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=True,
                        help='')
    parser.add_argument('--ALL_OBJ_ON_THE_GROUND',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=True,
                        help='')

    # ! default set False for contact loss.
    parser.add_argument('--CONTACT_MSE',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
                        

    # init scene model: orientation = Identity Matrix
    parser.add_argument('--IDENDITY_ROTATE_FOR_PROXD',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict
