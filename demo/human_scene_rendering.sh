#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
echo 'sciptpath:' $SCRIPTPATH
BASEDIR=$(dirname "$SCRIPTPATH")
# BASEDIR=$(dirname "$BASEDIR")
echo 'basedir:' $BASEDIR
cd $BASEDIR/demo
### end of dir path
cur_time=$(date +'%d.%H.%M.%S.%N')
img_list=-1
echo ${img_list}
echo ${cur_time}

### input information
input_dir=$1
batch_size=$2
stage3_kind_flag=$3
input_fps=$4
load_all_scene=True # set True to load final scene.
pare_dir=${input_dir}/smplifyx_results_st2_newCamera_gpConstraints_posaVersion
scene_init_fn=${input_dir}/refine_results/scene_reconstruction_s3kind1/obj_-1/model_scene_1_lr0.002_end.pth
save_dir=${input_dir}/refine_results/refined_scene_with_bodies_rendering
echo 'process' ${input_dir}

### preprocess input
scene_dir=Total3D_label_occNet_results/Total3D_input/
img_path=${input_dir}/${scene_dir}/img.jpg
img_dir_det=${input_dir}/Color_flip_rename_pointrend_X_101_det_all
img_dir=${input_dir}/Color_flip_rename
cam_inc_fn=${input_dir}/${scene_dir}/cam_K.txt
# smplify input
op_dir=${input_dir}/mv_smplifyx_input_OneEuroFilter_PARE_PARE3DJointOneConfidence_OP2DJoints
posa_dir=${input_dir}/smplifyx_results_st0/results/posa_contact_npy_newBottom
# 3D scene 
scene_result_path=${input_dir}/${scene_dir}
## new optimized camera parameters
CALIBRATION_FOLDER=${input_dir}/smplifyx_results_st2_newCamera_gpConstraints_posaVersion/smplifyx_cam/
scanned_path=${input_dir}/prox_scans_pre

## SMPL-X Model
MODEL_FOLDER=../data/smpl-x_model/models
VPOSER_FOLDER=../data/smpl-x_model/vposer_v1_0
PART_SEGM_FN=../data/smpl-x_model/smplx_parts_segm.pkl
### * enviornment
# source /home/hyi/venv/py3.6_hdsr/bin/activate
export LD_LIBRARY_PATH=/home/hyi/anaconda3/envs/pymesh_py3.6/lib:${LD_LIBRARY_PATH}
module load cuda/10.2
export RUN_PYTHON_PATH=/home/hyi/anaconda3/envs/mover_pt3d_new/bin/python
### * end of enviornment
${RUN_PYTHON_PATH} demo_refine_scene.py \
    --config ${BASEDIR}/config/fit_smplx.yaml \
    --img_path ${img_path} \
    --img_dir ${img_dir} \
    --img_dir_det ${img_dir_det} \
    --pare_dir ${pare_dir} \
    --posa_dir ${posa_dir} \
    --save_dir ${save_dir} \
    --pure_scene_loss True \
    --update_gp_camera False \
    --scene_visualize="False" \
    --debug="True" \
    --scene_result_path ${scene_result_path} \
    --ground_contact_path None \
    --cams_scalenet_fn None \
    --load_scalenet_cam=False \
    --scene_result_dir None \
    --load_all_scene ${load_all_scene} \
    --cam_inc_fn ${cam_inc_fn} \
    --start_stage 3 \
    --end_stage 3 \
    --scene_init_model ${scene_init_fn} \
    --img_list ${img_list} \
    --batch_size ${batch_size} \
    --data_folder ${op_dir} \
    --output_folder ${save_dir} \
    --visualize="False" \
    --pre_load="True" \
    --pre_load_pare_pose="False" \
    --scanned_path ${scanned_path} \
    --model_folder ${MODEL_FOLDER} \
    --vposer_ckpt ${VPOSER_FOLDER} \
    --part_segm_fn ${PART_SEGM_FN}  \
    --camera_type "user" \
    --use_body2scene_conf "True" \
    --gender male \
    --cluster="True" \
    --resample_in_sdf "True" \
    --calib_path ${CALIBRATION_FOLDER} \
    --calib_path_oriJ3d None \
    --start_opt_stage 3 \
    --end_opt_stage 5 \
    --stage3_idx 31 \
    --stage3_kind_flag ${stage3_kind_flag} \
    --input_obj_idx -1 \
    --input_fps ${input_fps} \
    --orientation_sample_num 1 \
    --use_total3d_reinit False \
    --preload_body True \
    --CONTACT_MSE False \
    --REINIT_SCALE_POSITION_BY_BODY True \
    --constraint_scale_for_chair True \
    --chair_scale 0.6\
    --only_rendering True \
    --ONLY_SAVE_FILETER_IMG False\