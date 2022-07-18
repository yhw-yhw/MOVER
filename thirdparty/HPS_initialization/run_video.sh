#!/bin/bash
# img_list=`expr "$1" + "2"`
BASEDIR=$(dirname "$0")
echo "run_sh: $BASEDIR"
img_list=-1
INPUT_DIR=$1
batch_size=$2
INPUT_DATA="${INPUT_DIR}/smplifyx_video_input/"
DATA_FOLDER=${INPUT_DIR}/"mv_smplifyx_input_OneEuroFilter_PARE_PARE3DJointOneConfidence_OP2DJoints"
OUTPUT_FOLDER=${INPUT_DATA}/"results_debug"
CALIBRATION_FOLDER=${INPUT_DATA}/smplifyx_cam
CONFIG_FILE=${BASEDIR}/../body_models/cfg_files/fit_smplx_video.yaml
echo ${DATA_FOLDER}
echo ${OUTPUT_FOLDER}
MODEL_FOLDER=${BASEDIR}/../../data/smpl-x_model/models
VPOSER_FOLDER=${BASEDIR}/../../data/smpl-x_model/vposer_v1_0
part_segm_fn_path=${BASEDIR}/../../data/smpl-x_model/smplx_parts_segm.pkl
# source /is/cluster/hyi/venv/py3.6_hdsr/bin/activate
# /home/hyi/anaconda3/envs/mover_pt3d_new
# ! save_meshes=True: save mesh and rendered images.
python HPS_initialization/main.py \
    --single "False" \
    --config ${CONFIG_FILE} \
    --img_list ${img_list} \
    --batch_size ${batch_size} \
    --data_folder ${DATA_FOLDER} \
    --output_folder ${OUTPUT_FOLDER} \
    --visualize="False" \
    --save_meshes=True \
    --model_folder ${MODEL_FOLDER} \
    --model_type 'smplx' \
    --pre_load="False" \
    --pre_load_pare_pose="False" \
    --vposer_ckpt ${VPOSER_FOLDER} \
    --part_segm_fn ${part_segm_fn_path} \
    --camera_type "user" \
    --gender 'male' \
    --use_video "True" \
    --calib_path ${CALIBRATION_FOLDER} \
    --start_opt_stage 3 \
    --end_opt_stage 5 \