DIR=$1
VARIABLE=${DIR}
DATASET=$2
echo ${DATASET}
if [ "${DATASET}" = "piGraph" ];
then
    echo "process not prox: "${DATASET}
    cp ${DIR}/../../cam_K.txt ${DIR}/cam_K.txt
    cp ${DIR}/../Color_flip_rename_pointrend_X_101_det_all/000001/img.jpg ${DIR}
    cp ${DIR}/../Color_flip_rename_pointrend_X_101_det_all/000001/detections.json ${DIR}
elif [ "${DATASET}" = "i3DB" ];
then
    echo "process not prox: "${DATASET}
    cp ${DIR}/../../cam_K.txt ${DIR}/cam_K.txt
    cp ${DIR}/../Color_flip_rename_pointrend_X_101_det_all/000001/img.jpg ${DIR}
    cp ${DIR}/../Color_flip_rename_pointrend_X_101_det_all/000001/detections.json ${DIR}

else
    echo "process prox"
fi
sleep 1s
# cd /is/cluster/hyi/workspace/HCI/Total3DUnderstanding/
# source /is/cluster/hyi/workspace/HCI/Total3DUnderstanding/source.sh
# source activate Total3D
# echo ${VARIABLE}
python main.py configs/total3d.yaml --mode demo --demo_path ${VARIABLE} 
