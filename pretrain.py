# Set the arguments
import argparse
import random
import torch
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,  default=1445, help='random seed')
parser.add_argument('--data', type=str, default='../../../data/CHAOS_png/Dataset')
parser.add_argument('--save_path', type=str, default='checkpoint/Denoise/')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--resume_path', type=str, default='best_model/UNet_CHAOS_val_loss.pth')
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

from torch.utils.tensorboard import SummaryWriter
from model.unet_model import UNet
from utils.dataset import Denoise_Loader
from utils.loss import DiceLoss
from utils.pytorchtools import EarlyStopping
from torch import optim
from tqdm import tqdm
import torch.nn as nn
from torchvision import utils


def train(train_loader, optimizer, fn_REG, epoch, writer):
    loss_list = []
    net.train()
    with tqdm(enumerate(train_loader), unit="batch") as tepoch:
        for batch, data in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")

            label = data['label'].to(device, dtype=torch.float32)
            input = data['input'].to(device, dtype=torch.float32)
            mask = data['mask'].to(device, dtype=torch.uint8)

            output = net(input)

            optimizer.zero_grad()

            loss_G = fn_REG(output * (1 - mask), label * (1 - mask))

            loss_list.append(loss_G.item())

            tepoch.set_postfix(loss=loss_G.item())

            loss_G.backward()
            optimizer.step()

    # write to tensorboard
    writer.add_scalar('train_loss/loss', np.mean(loss_list), global_step=epoch)

    return np.mean(loss_list)

def validate(val_loader, fn_REG, epoch, writer):
    val_loss_list = []
    with tqdm(enumerate(val_loader), unit="batch") as tepoch:
        for batch, data in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")

            label = data['label'].to(device, dtype=torch.float32)
            input = data['input'].to(device, dtype=torch.float32)
            mask = data['mask'].to(device, dtype=torch.uint8)

            output = net(input)

            loss_G = fn_REG(output * (1 - mask), label * (1 - mask))

            val_loss_list.append(loss_G.item())

            tepoch.set_postfix(loss=loss_G.item())

    # write to tensorboard
    writer.add_scalar('val_loss/loss', np.mean(val_loss_list), global_step=epoch)
    return np.mean(val_loss_list)

def train_net(net, device, data_path, save_path, epochs=200, batch_size=4, lr=1e-4):

    os.makedirs(save_path, exist_ok=True)
    # Training set
    train_dataset = Denoise_Loader(data_path, val_mode=False)
    val_dataset = Denoise_Loader(data_path, val_mode=True)

    print("The num of training data：", train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)
    print("The num of validating data：", val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=False)
    # Adam
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9,0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                          verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    early_stopping = EarlyStopping(patience=20, verbose=True)
    # MSE loss
    fn_REG = nn.L1Loss().to(device)
    # best_loss
    best_loss = float('inf')
    writer = SummaryWriter(log_dir=save_path + 'logs')
    # Training and validating each epoch
    for epoch in range(epochs):
        loss = train(train_loader, optimizer, fn_REG, epoch, writer)
        print('loss=%.5f'%(loss))
        val_loss = validate(val_loader, fn_REG, epoch, writer)
        print('val_loss=%.5f'%(val_loss))
        scheduler.step(val_loss)
        # Save the model with the smallest val_loss
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(save_path + 'best_model', exist_ok=True)
            torch.save(net.state_dict(), save_path + 'best_model/CHAOS_val_loss.pth')
            print('yeah! save!')
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print('\nbest_loss=%.5f'%(best_loss))


if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)

    if os.path.exists(args.resume_path):
        path_checkpoint = 'best_model/UNet_CHAOS_val_loss.pth'
        print('Resume training from', path_checkpoint)
        checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint)

    train_net(net, device, args.data, args.save_path, epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr)
