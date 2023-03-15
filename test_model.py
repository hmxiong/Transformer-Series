# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

# import datasets
# import util.misc as utils
# from datasets import build_dataset, get_coco_api_from_dataset
# from engine import evaluate, train_one_epoch
# from models import build_model 
from model.position_encoding import PositionEmbeddingSine

def main(): 
    a = np.ones([2, 256, 228, 228], dtype=float)
    d = torch.tensor(a)
    position_embedding = PositionEmbeddingSine(256 // 2, normalize=True)
    out = position_embedding(d)
    print(out.shape)


if __name__ == '__main__':
    main()
