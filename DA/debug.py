import os
import sys

sys.path.extend(['./models', './data_loader'])
import torch
import logging
import importlib
import numpy as np
from datetime import datetime
import utils
from opt import parse_args

import os

data_root = os.path.join(args.data_dir,'source')
print(f"Checking data_root: {data_root}")
if not os.path.exists(data_root):
    print("Error: data_root does NOT exist!")

files = os.listdir(data_root)
print(f"Number of files in {data_root}: {len(files)}")
print("Some example files:", files[:5])

