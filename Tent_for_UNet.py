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
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-6)
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
from model.unet_model import UNet
from utils.dataset import Test_UNet_Loader
import torch.nn as nn

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # return -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return -(x * torch.log(x)).sum([2,3])

def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

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

    def predict(self, adaptnet, train_test_dataset, show_mask_path, batch_size=1, average=False):

        adaptnet.eval()
        self.train_test_loader = torch.utils.data.DataLoader(dataset=train_test_dataset,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             num_workers=4,
                                                             pin_memory=True)
        for data in tqdm(self.train_test_loader):
            input = data['input'].to(device, dtype=torch.float32)
            input2 = data['input2'].to(device, dtype=torch.float32)
            name = data['name']
            # Predict
            pred_mask, denoise = adaptnet(input)
            pred_mask = np.array(pred_mask.data.cpu()[0])[0]
            if average:
                pred_mask2, _ = adaptnet(input2)
                pred_mask2 = np.array(pred_mask2.data.cpu()[0])[0]

            if average:
                pred_mask[(pred_mask + pred_mask2) / 2 >= 0.5] = 255
                pred_mask[(pred_mask + pred_mask2) / 2 < 0.5] = 0
            else:
                pred_mask[pred_mask >= 0.5] = 255
                pred_mask[pred_mask < 0.5] = 0

            cv2.imwrite(os.path.join(show_mask_path, name[0].split('/')[-1]), pred_mask)

        print(len(train_test_dataset))
        print("Predict complete!")

    def train(self,original_path,train_test_dataset,net,step,lr,batch_size=8):
        # Load the model
        net.load_state_dict(torch.load(original_path, map_location=device))
        print("The num of the target data：", train_test_dataset.__len__())
        self.test_loader = torch.utils.data.DataLoader(dataset=train_test_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=4,
                                                        pin_memory=True,
                                                        drop_last=True)

        for param in adaptnet.parameters():
            param.requires_grad = False
        # BN layers
        parameters, names = collect_params(adaptnet)
        for param in parameters:
            param.requires_grad = True

        if step == 0:
            print('no train!')
        else:

            optimizer = optim.Adam(parameters, lr=lr, betas=(0.9, 0.99))
            for _ in range(step):

                net.train()
                optimizer.zero_grad()
                for inputs in tqdm(self.train_test_loader, desc='train %s' % (test_path.split('/')[-1])):
                    inputs = inputs.to(device=device, dtype=torch.float32)
                    forward_and_adapt(inputs, net, optimizer)

        return net


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    step = args.step
    lr = args.lr
    original_path = args.model
    mode_list = args.mode_list
    data_dir = args.data_dir
    # The root of the model, as well as the save_path
    root_path = '/'.join(original_path.split('/')[:-2]) + '/'
    print(root_path)

    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)

    save_name = '_Tent'
    for mode in mode_list:
        print(mode)
        if mode == 'train':
            tests_path = data_dir + '/CHAOS_png/Train_Sets/CT'
            shows_path = root_path + 'Result/optimize_batch%s/%d/CHAOS_train' % (save_name, step)
        elif mode == 'test':
            tests_path = data_dir + '/CHAOS_png/Test/image'
            shows_path = root_path + 'Result/optimize_batch%s/%d/CHAOS_test' % (save_name, step)
        elif mode == 'LITS':
            tests_path = data_dir + '/LITSChallenge/image'
            shows_path = root_path + 'Result/optimize_batch%s/%d/LITS' % (save_name, step)
        for root, dirs, files in os.walk(tests_path):
            if files:
                test_path = os.path.join(tests_path, root.split('/')[-1])
                show_mask_path = os.path.join(shows_path+'/mask', root.split('/')[-1])
                os.makedirs(show_mask_path, exist_ok=True)
                mydata = Show(1)
                train_test_dataset = Test_UNet_Loader(test_path)
                adaptnet = mydata.train(original_path, train_test_dataset, net, step, lr, batch_size=8)
                mydata.predict(adaptnet, train_test_dataset, show_mask_path, average=args.average)