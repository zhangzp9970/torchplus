import random
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def init_seed(seed:int)->None:
    cudnn.benchmark = True
    cudnn.deterministic = True
    seed = 9970
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def auto_save_file(filename,target_dir)->None:
    with open(os.path.join(target_dir, os.path.basename(filename)), 'w', encoding='utf-8') as fout:
        with open(os.path.abspath(filename), 'r', encoding='utf-8') as fin:
            file = fin.read()
            fout.write(file)
