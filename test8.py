import os
import sys
import random
import shutil
import logging
import argparse

from tqdm import tqdm, trange

import numpy as np

from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

model = torch.nn.Linear(10, 10)
parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
print(model.parameters())
print(parameters)







