# Ground Contact Detector

## Data Preparation
Make soft links
```
ln -s $DATAPATH$/WACV2020_data/contact_dataset data/contact_dataset
ln -s $DATAPATH$/WACV2020_data/Human3.6M data/Human3.6M
ln -s $DATAPATH$/WACV2020_data/MADS data/MADS
```

## Dependency
This code was developed with Python3.6, PyTorch-1.0.1, cuda-9.0, cudnn-9.0-v7.4 in Unbuntu16.04.

## (Optional) Python packages
To use tensorboard to record training curve, you need to install `TensorFlow`(any version after 1.5, even cpu-only is okay) and `tensorboardX`.

## Inference
You can use the pre-trained models to do inference for a video
```
python inference.py
```

## Training
```
python train.py
```

You can specify different options in the command line, please check the code.

## Test on the labeled dataset
After you have trained your model, you can test it on the contact dataset.
```
python test.py
```

