## Getting started

Start by cloning the repo:

```bash
git clone 
cd MOVER
```

## Environment

- Ubuntu 20 / 18
- **CUDA=10.2, GPU Memory > 4GB**
- Python = 3.8
- PyTorch = 1.7.1 
- PyTorch3D = 0.6.1 (official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) )

```bash
# install conda, skip if already have
# wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
# chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
# bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -f -p /usr/local
# rm Miniconda3-py38_4.10.3-Linux-x86_64.sh

conda config --env --set always_yes true
conda update -n base -c defaults conda -y

# Note: choose one of them to install
# Install by conda_install.sh
bash conda_install.sh

# Install by 'environment.yaml' 
conda env create -f environment.yaml
conda init bash
source ~/.bashrc
source activate mover
pip install -r requirements.txt 
```