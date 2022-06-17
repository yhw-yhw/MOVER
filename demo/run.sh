### change the path to where you extract the demo dataset.
DATA_DIR=/is/cluster/work/hyi/dataset/MOVER_dataset
### run #1 PROX qualitative
# INPUT_DIR=${DATA_DIR}/PROX_qualitative/MPH8_00034_01
# ./scene_refinement.sh ${INPUT_DIR} 2949 1 30
### run #2 PROX quantitative
INPUT_DIR=${DATA_DIR}/PROX_quantitative
./scene_refinement.sh ${INPUT_DIR} 178 0 1
### run #3 piGraph
# INPUT_DIR=${DATA_DIR}/piGraph/3bHallway_mati2_2014-04-30-22-46-22
# ./scene_refinement.sh ${INPUT_DIR} 849 1 5