BASEDIR=$(dirname "$0")
echo "run_sh: $BASEDIR"

# module load cuda/10.2
conda create -n mover python=3.8 -y
conda activate mover

# conda install -c creditx gcc-7 -y
source ~/export_gcc.sh
module load cuda/10.2

# in workstation
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch

# install pytorch3d
export FORCE_CUDA=1
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# conda install pytorch3d -c pytorch3d

# both from multipersons: https://github.com/JiangWenPL/multiperson
# build neural render
cd ${BASEDIR}/modules/neural_renderer
python setup.py install

# [option] neural render farthest_depth version to get the farthest depth of the surface
# cd ${BASEDIR}/modules/neural_renderer_farthest_depth
# python setup.py install

# psmesh: https://github.com/MPI-IS/mesh
cd ${BASEDIR}/modules/mesh
make all

# ChamferDistancePytorch
cd ${BASEDIR}/modules/ChamferDistancePytorch
python setup.py install

# pip install
pip install -r requirements.txt --use-deprecated=legacy-resolver

# install sdf calculation
