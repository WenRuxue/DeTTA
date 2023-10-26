# Set the arguments
import argparse
import random
import torch
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,  default=1445, help='random seed')
parser.add_argument('--data', type=str, default='../../../data/CHAOS_png/Dataset')
parser.add_argument('--save_path', type=str, default='checkpoint/DeYNet')
parser.add_argument('--pretrain_path', type=str, default='../Denoise/best_model/CHAOS_val_loss.pth')
parser.add_argument('--pretrain_part', type=list, default=['decoder_seg'],help='0-3 parts of encoder/decoder_seg/decoder_de')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_val', type=float, default=30., help='The max value of w(t)')
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
from model.unet_model import DeYNet
from utils.dataset import DeYNet_loader
from utils.loss import DiceLoss
from utils.pytorchtools import EarlyStopping
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from itertools import cycle

def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


def weight_schedule(epoch, max_epochs, max_val, mult, n_labeled, n_samples):
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epochs, max_val, mult)

def train(train_label_loader, train_unlabel_loader, optimizer, Dice, fn_REG, epoch, writer, k=2500, max_epochs=200,
          max_val=30., ramp_up_mult=-5., n_samples=6033):

    label_loss_list,unlabel_loss_list,loss_list = [],[],[]
    net.train()
    # evaluate unsupervised cost weight
    w = weight_schedule(epoch, max_epochs, max_val, ramp_up_mult, k, n_samples)
    print('unsupervised loss weight : {}'.format(w))
    # turn it into a usable pytorch object
    w = torch.autograd.Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)
    pbar = tqdm(total=len(train_unlabel_loader),position=0,colour='blue',leave=True,ncols=80)
    for batch,data in enumerate(zip(cycle(train_label_loader),train_unlabel_loader)):
        pbar.set_description(f"Epoch {epoch + 1}")
        # label_data
        label = data[0]['label'].to(device, dtype=torch.float32)
        input = data[0]['input'].to(device, dtype=torch.float32)
        mask = data[0]['mask'].to(device)
        seg_label = data[0]['seg_label'].to(device)
        # unlabel_data
        label_ = data[1]['label'].to(device, dtype=torch.float32)
        input_ = data[1]['input'].to(device, dtype=torch.float32)
        mask_ = data[1]['mask'].to(device)

        seg_pred, output = net(input)
        _, output_ = net(input_)

        optimizer.zero_grad()

        # Denoising loss
        loss_D = fn_REG(output * (1 - mask), label * (1 - mask))
        loss_D_ = fn_REG(output_ * (1 - mask_), label_ * (1 - mask_))
        unlabel_loss = (loss_D + loss_D_)/2
        # Segmentation loss
        label_loss = Dice(seg_pred, seg_label)
        # Combline losses
        loss = label_loss + w * unlabel_loss

        loss_list.append(loss.item())
        label_loss_list.append(label_loss.item())
        unlabel_loss_list.append(unlabel_loss.item())

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


def train_net(net, device, data_path, save_path, epochs=200, batch_size=4, lr=1e-4, max_val=30.):

    os.makedirs(save_path, exist_ok=True)
    # Training set
    train_label_dataset = DeYNet_loader(data_path)
    train_unlabel_dataset = DeYNet_loader(data_path, label_mode=False)
    val_dataset = DeYNet_loader(data_path, val_mode=True)

    print("The num of labeled training data：", train_label_dataset.__len__())
    train_label_loader = torch.utils.data.DataLoader(dataset=train_label_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     num_workers=4,
                                                     pin_memory=True)
    print("The num of unlabeled training data：", train_unlabel_dataset.__len__())
    train_unlabel_loader = torch.utils.data.DataLoader(dataset=train_unlabel_dataset,
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
    # Noise2Void loss (MSE loss)
    fn_REG = nn.L1Loss().to(device)
    # Segmentation loss
    Dice = DiceLoss()
    # best_loss
    best_loss = float('inf')
    writer = SummaryWriter(log_dir=save_path+'logs')
    # Training and validating each epoch
    for epoch in range(epochs):
        # train the labeled and unlabeled data
        loss = train(train_label_loader, train_unlabel_loader, optimizer, Dice, fn_REG, epoch, writer, k = len(train_label_dataset), n_samples = len(train_label_dataset)+len(train_unlabel_dataset),
                     max_epochs=epochs,max_val=max_val, ramp_up_mult=-5.)
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


def pretrain(net, pretrain_path, pretrain_part):
    print('load pretrained model(DeNoise)', pretrain_path, 'for', pretrain_part)
    pretrained_dict = torch.load(pretrain_path)
    two_stream_dict = net.state_dict()
    for k, v in two_stream_dict.items():
        # down
        if k in pretrained_dict:
            if 'encoder' in args.pretrain_part:
                two_stream_dict[k] = pretrained_dict[k]
            else:
                pass
        # up
        else:
            if 'decoder_seg' in args.pretrain_part and 'decoder_de' in args.pretrain_part:
                k_0 = k.replace('_mask', '').replace('_de', '')
            elif 'decoder_de' in args.pretrain_part:
                k_0 = k.replace('_de', '')
            elif 'decoder_seg' in args.pretrain_part:
                k_0 = k.replace('_mask', '')
            else:
                pass
            if k_0 in pretrained_dict:
                two_stream_dict[k] = pretrained_dict[k_0]

    net.load_state_dict(two_stream_dict)
    return net


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = DeYNet(n_channels=1, n_classes=1)
    net.to(device=device)

    if os.path.exists(args.resume_path):
        path_checkpoint = 'best_model/UNet_CHAOS_val_loss.pth'
        print('Resume training from',path_checkpoint)
        checkpoint = torch.load(path_checkpoint, map_location = torch.device('cpu'))
        net.load_state_dict(checkpoint)

    if os.path.exists(args.pretrain_path):
        net = pretrain(net, args.pretrain_path, args.pretrain_part)
    else:
        print('The pretrain_path does not exist! No pretrain is done.')

    train_net(net, device, args.data, args.save_path, epochs=args.n_epochs, batch_size=args.batch_size//2, lr=args.lr, max_val=args.max_val)