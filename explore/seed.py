# set seed everything 
# numpy, torch 

import random
import numpy as np
import os
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


"""
src.seed import 진행한 이후 
seed_everything(42) 이렇게 설정해서 진행 
"""