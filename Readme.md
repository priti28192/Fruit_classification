The aim of this code is to classify fruits as damaged or not i.e. split or no split using convolutional neural network.

## Overview 
To train the code, please use the following

python train.py --dataroot path/to/data


The following options can be passed to train.py

  --name = Experiment name
  --num_workers = Number of data loading workers
  --rdm_seed = Random seed
  --device = Name of device to use for tensor computations (cuda/cpu)
  --display_step = Interval in batches between display of training metrics
  --epochs = Number of epochs 
  --batch_size = Batch size
  --lr = Learning rate
  --momentum = Momentum for Adam optimizer
