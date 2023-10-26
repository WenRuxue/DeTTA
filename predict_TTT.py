# Set the arguments
import argparse
import random
import torch
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,  default=1445, help='random seed')
parser.add_argument('--model', type=str, default='checkpoint/UNet/best_model/CHAOS_val_loss.pth')
parser.add_argument('--mode_list', nargs='+', type=str, default=['test', 'LITS'])
parser.add_argument('--data_dir', type=str, default='../../../data')
parser.add_argument('--average', action="store_true", help='average or not')
args = parser.parse_args()

# Set the random seed at the beginning
def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
setup_seed(args.seed)

import copy
from torch import optim
import sys
import cv2
from tqdm import tqdm
sys.path.append('../')
from model.unet_model import UNet_rotation
from utils.dataset import Test_UNet_Loader
import torch.nn as nn

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


class Show(object):
    
    def __init__(self, m, n = 1):

        self.m = m
        self.n = n

    def predict(self, original_path, train_test_dataset, show_mask_path, batch_size=1):

        # Load the model
        net.load_state_dict(torch.load(original_path, map_location=device))
        net.eval()
        self.train_test_loader = torch.utils.data.DataLoader(dataset=train_test_dataset,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             num_workers=4,
                                                             pin_memory=True)
        for data in tqdm(self.train_test_loader):
            input = data['image'].to(device, dtype=torch.float32)
            name = data['name']
            # Predict
            pred_mask,_ = net(input)
            pred_mask = np.array(pred_mask.data.cpu()[0])[0]

            pred_mask[pred_mask >= 0.5] = 255
            pred_mask[pred_mask < 0.5] = 0

            cv2.imwrite(os.path.join(show_mask_path, name[0].split('/')[-1]), pred_mask)

        print(len(train_test_dataset))
        print("Predict complete!")


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_path = args.model
    mode_list = args.mode_list
    data_dir = args.data_dir
    # The root of the model, as well as the save_path
    root_path = '/'.join(original_path.split('/')[:-2]) + '/'
    print(root_path)

    net = UNet_rotation(n_channels=1, n_classes=1)
    net.to(device=device)

    for mode in mode_list:
        print(mode)
        if mode == 'train':
            tests_path = data_dir + '/CHAOS_png/Train_Sets/CT'
            shows_path = root_path + 'Result/no_optimize/CHAOS_train'
        elif mode == 'test':
            tests_path = data_dir + '/CHAOS_png/Test/image'
            shows_path = root_path + 'Result/no_optimize/CHAOS_test'
        elif mode == 'LITS':
            tests_path = data_dir + '/LITSChallenge/image'
            shows_path = root_path + 'Result/no_optimize/LITS'
        for root, dirs, files in os.walk(tests_path):
            if files:
                test_path = os.path.join(tests_path, root.split('/')[-1])
                show_mask_path = os.path.join(shows_path+'/mask', root.split('/')[-1])
                os.makedirs(show_mask_path, exist_ok=True)
                mydata = Show(1)
                train_test_dataset = Test_UNet_Loader(test_path)
                mydata.predict(original_path, train_test_dataset, show_mask_path)