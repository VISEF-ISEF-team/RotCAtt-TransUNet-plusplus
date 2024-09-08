import argparse
import numpy as np
import nibabel as nib
from skimage.transform import resize as skires
import csv
import yaml
import numpy as np

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
class DotDict:
    def __init__(self, dictionary):
        self._dict = dictionary

    def __getattr__(self, attr):
        value = self._dict[attr]
        if isinstance(value, dict):
            return DotDict(value)
        return value
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default=None, help='architecture name')
    parser.add_argument('--name', default=None, help='model name')
    parser = parser.parse_args()
    with open(f'outputs/{parser.network}/{parser.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotDict(config)
    return config
        
def save_vol(vol, type, path):
    vol = np.transpose(vol, (2, 1, 0))
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(vol.astype(np.int8), affine) if type=='labels' else nib.Nifti1Image(vol, affine)
    nib.save(nifti_file, path)    
    
def resize_vol(vol, new_size):
    return skires(vol, new_size, order=1, preserve_range=True, anti_aliasing=False)


def write_csv(path, data):
    with open(path, mode='a', newline='') as file:
        iteration = csv.writer(file)
        iteration.writerow(data)
    file.close()
    

