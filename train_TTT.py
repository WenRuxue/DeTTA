# Set the arguments
import argparse
import random
import torch
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,  default=1445, help='random seed')
parser.add_argument('--data', type=str, default='../../../data/CHAOS_png/Dataset')
parser.add_argument('--save_path', type=str, default='checkpoint/TTT')
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
import sys
sys.path.append('../')
from model.unet_model import UNet_rotation
from utils.dataset import UNet_Loader
from utils.loss import DiceLoss
from utils.pytorchtools import EarlyStopping
from utils.rotation import rotate_batch
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable

def temporal_loss(image, label, criterion):
    # unlabel_loss: rotation
    inputs_ssh, labels_ssh = rotate_batch(image, 'rand')
    _, outputs_ssh = net(inputs_ssh)
    labels_ssh = labels_ssh.to(device=device)
    # label_loss
    predict_mask, _ = net(image)
    def masked_dice(out, labels=None, label_flag=False):
        if label_flag:
            dice_loss = criterion(out, labels)
            return dice_loss
        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False)

    sup_loss = masked_dice(predict_mask, label, label_flag=True)
    cross = nn.CrossEntropyLoss().cuda()
    unsup_loss = cross(outputs_ssh, labels_ssh)

    return sup_loss + unsup_loss, sup_loss, unsup_loss

def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


def weight_schedule(epoch, max_epochs, max_val, mult, n_labeled, n_samples):
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epochs, max_val, mult)

def train(train_label_loader, optimizer, Dice, epoch, writer):

    label_loss_list,unlabel_loss_list,loss_list = [],[],[]
    net.train()
    pbar = tqdm(total=len(train_label_loader), position=0, colour='blue', leave=True, ncols=80)
    for i, data in tqdm(enumerate(train_label_loader), unit="batch"):
        pbar.set_description(f"Epoch {epoch + 1}")
        # label_data
        image = data[0]
        label = data[1]
        optimizer.zero_grad()

        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        loss, label_loss, unlabel_loss = temporal_loss(image, label, Dice)

        label_loss_list.append(label_loss.item())
        unlabel_loss_list.append(unlabel_loss.item())
        loss_list.append(loss.item())

        pbar.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        pbar.update(1)

    pbar.close()
    # write to tensorboard
    writer.add_scalar('train_loss/loss', np.mean(loss_list), global_step=epoch)

    return np.mean(loss_list)

def validate(val_loader, Dice, epoch, writer):
    net.eval()
    with torch.no_grad():
        val_loss_list = []
        pbar = tqdm(total=len(val_loader), position=0, colour='blue', leave=True, ncols=80)
        # with enumerate(val_loader) as tepoch:
        for batch, data in enumerate(val_loader):
            pbar.set_description(f"Epoch {epoch + 1}")
            input = data['input'].to(device, dtype=torch.float32)
            seg_label = data['seg_label'].to(device)
            # Validate
            seg_pred, _ = net(input)
            # Segmentation loss
            loss = Dice(seg_pred, seg_label)
            val_loss_list.append(loss.item())

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

        pbar.close()
    # write to tensorboard
    writer.add_scalar('val_loss/loss', np.mean(val_loss_list), global_step=epoch)

    return np.mean(val_loss_list)


def train_net(net, device, data_path, save_path, epochs=200, batch_size=4, lr=1e-4):

    data_path = os.path.join(data_path, 'label_data')
    os.makedirs(save_path, exist_ok=True)
    # Training set
    train_label_dataset = UNet_Loader(data_path)
    val_dataset = UNet_Loader(data_path, val_mode=True)

    print("The num of labeled training data：", train_label_dataset.__len__())
    train_label_loader = torch.utils.data.DataLoader(dataset=train_label_dataset,
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
    # Segmentation loss
    Dice = DiceLoss()
    # best_loss
    best_loss = float('inf')
    writer = SummaryWriter(log_dir=save_path+'logs')
    # Training and validating each epoch
    for epoch in range(epochs):
        # train the labeled and unlabeled data
        loss = train(train_label_loader, optimizer, Dice, epoch, writer)
        print('loss=%.5f'%(loss))
        # Validating
        val_loss = validate(val_loader, Dice, epoch, writer)
        print('val_loss=%.5f'%(val_loss))

        scheduler.step(val_loss)
        # Save the model with the smallest val_loss
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(save_path+'best_model',exist_ok=True)
            torch.save(net.state_dict(), save_path+'best_model/CHAOS_val_loss.pth')

        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('\nbest_loss=%.5f'%(best_loss))



if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet_rotation(n_channels=1, n_classes=1)
    net.to(device=device)

    if os.path.exists(args.resume_path):
        path_checkpoint = 'best_model/UNet_CHAOS_val_loss.pth'
        print('Resume training from',path_checkpoint)
        checkpoint = torch.load(path_checkpoint, map_location = torch.device('cpu'))
        net.load_state_dict(checkpoint)

    train_net(net, device, args.data, args.save_path, epochs=args.n_epochs, batch_size=args.batch_size//2, lr=args.lr)