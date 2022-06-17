import itertools
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import cv2
import pickle
from loguru import logger
import trimesh
from torch.nn.functional import smooth_l1_loss, mse_loss, l1_loss
# import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from ..constants import (
    DEBUG_LOSS, DEBUG_LOSS_OUTPUT,
    DEBUG_DEPTH_LOSS,
    DEBUG_CONTACT_LOSS,
    DEBUG,
    USE_HAND_CONTACT_SPLIT,
    BBOX_HEIGHT_CONSTRAINTS,
)

import neural_renderer as nr
# import neural_renderer_farthest_depth as nr_fd
