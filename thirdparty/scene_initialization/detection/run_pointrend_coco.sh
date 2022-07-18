#!/bin/bash
DIR=$1
echo ${DIR}
cd /is/cluster/hyi/workspace/HCI/tool_scripts/detection/
/home/hyi/anaconda3/envs/pt1.7_cu10.1_newestDetron2/bin/python test_det_pointrend.py \
	--img-root=${DIR} \
    --out-img-root=${DIR}_pointrend_X_101_det_all \
	--width=1920 \
	--height=1080 \
	--bbox-thr 0.55 \
	--save \
	--show \
	--idx=$2
