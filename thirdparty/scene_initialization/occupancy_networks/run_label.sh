#!/bin/bash
export PATH=$PATH:/usr/bin/
export PATH=$PATH:/home/hyi/anaconda3/bin
module load cuda/10.0
source activate mesh_funcspace
cd /is/cluster/hyi/workspace/Multi-IOI/occupancy_networks/
echo $1
/is/cluster/hyi/anaconda3/envs/mesh_funcspace/bin/python pre_post.py $1 'pre' 'Total3D_label'
/is/cluster/hyi/anaconda3/envs/mesh_funcspace/bin/python generate.py $1/Total3D_label_occNet_input/demo_obj.yaml
/is/cluster/hyi/anaconda3/envs/mesh_funcspace/bin/python pre_post.py $1 'post' 'Total3D_label'
