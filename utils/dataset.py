import copy
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class Denoise_Loader(Dataset):
    def __init__(self, data_path, val_mode=False, sgm=25, size_data=(1, 512, 512), ratio=0.9, size_window=(5, 5)):

        self.data_path = data_path
        self.sgm = sgm
        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window

        if val_mode:
            self.idx_list = np.loadtxt(os.path.join(self.data_path, 'label_data/val_full_list2.txt'),
                                           dtype=str)
            self.idx_list = ['label_data/image/' + i for i in self.idx_list]
        else:
            self.idx_list1 = np.loadtxt(os.path.join(self.data_path, 'label_data/train_full_list2.txt'),
                                               dtype=str)
            self.idx_list2 = np.loadtxt(os.path.join(self.data_path, 'unlabel_data/unlabel_list.txt'),
                                       dtype=str)
            self.idx_list = ['label_data/image/' + i for i in self.idx_list1] \
                            + ['unlabel_data/image/' + i for i in self.idx_list2]


    def __getitem__(self, index):

        image = Image.open(os.path.join(self.data_path, self.idx_list[index]))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.array(image)
        image = image.reshape(1, image.shape[0], image.shape[1])

        label = image
        input, mask = self.generate_mask(copy.deepcopy(label))

        data = {'label': label, 'input': input, 'mask': mask}
        return data

    def __len__(self):
        return len(self.idx_list)

    def generate_mask(self, input):

        ratio = self.ratio
        size_window = self.size_window
        size_data = self.size_data
        num_sample = int(size_data[2] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[0]):
            idy_msk = np.random.randint(0, size_data[2], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[2] - (idy_msk_neigh >= size_data[2]) * size_data[2]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (ich, idy_msk, idx_msk)
            id_msk_neigh = (ich, idy_msk_neigh, idx_msk_neigh)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask

class UNet_Loader(Dataset):
    def __init__(self, data_path, val_mode=False, label_mode=True):

        self.data_path = data_path
        self.val_mode = val_mode
        self.label_mode = label_mode
        if self.label_mode:
            if self.val_mode:
                self.idx_list = np.loadtxt(os.path.join(self.data_path, 'val_full_list2.txt'),
                                           dtype=str)
            else:
                self.idx_list = np.loadtxt(os.path.join(self.data_path, 'train_full_list2.txt'),
                                           dtype=str)
            self.mask_path = os.path.join(self.data_path, 'label')
        else:
            self.idx_list = np.loadtxt(os.path.join(self.data_path, 'unlabel_list.txt'), dtype=str)

        self.img_path = os.path.join(self.data_path, 'image')

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_path, self.idx_list[index]))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.array(image)
        image = image.reshape(1, image.shape[0], image.shape[1])
        if self.label_mode:
            label = Image.open(os.path.join(self.mask_path, self.idx_list[index]))
            label = np.array(label)
            label = label.reshape(1, label.shape[0], label.shape[1])
            return image, label
        else:
            return image

    def __len__(self):

        return len(self.idx_list)

class Test_UNet_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.idx_list = []
        for root,dir,files in os.walk(self.data_path):
            if files:
                for file in files:
                    self.idx_list.append(os.path.join(root,file))

    def __getitem__(self, index):
        image = Image.open(self.idx_list[index])
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.array(image)
        image = image.reshape(1, image.shape[0], image.shape[1])
        data = {'image':image,'name': self.idx_list[index]}
        return data

    def __len__(self):

        return len(self.idx_list)

class DeYNet_loader(Dataset):
    def __init__(self, data_path, label_mode=True, val_mode=False, sgm=25, size_data=(1, 512, 512), ratio=0.9, size_window=(5, 5)):
        self.data_path = data_path
        self.label_mode = label_mode
        self.val_mode = val_mode
        self.sgm = sgm
        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window

        if self.label_mode:
            if self.val_mode:
                self.idx_list = np.loadtxt(os.path.join(self.data_path, 'label_data/val_full_list.txt'),
                                               dtype=str)
            else:
                self.idx_list = np.loadtxt(os.path.join(self.data_path, 'label_data/train_full_list.txt'),
                                                   dtype=str)
            self.idx_list = ['label_data/image/' + i for i in self.idx_list]
        else:
            self.idx_list = np.loadtxt(os.path.join(self.data_path, 'unlabel_data/unlabel_list.txt'),
                                           dtype=str)
            self.idx_list = ['unlabel_data/image/' + i for i in self.idx_list]

    def __getitem__(self, index):

        image = Image.open(os.path.join(self.data_path, self.idx_list[index]))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.array(image)
        image = image.reshape(1, image.shape[0], image.shape[1])

        label = image
        input, mask = self.generate_mask(copy.deepcopy(label))

        if self.label_mode:
            seg_label = Image.open(os.path.join(self.data_path, self.idx_list[index].replace('image','label')))
            seg_label = np.array(seg_label)
            seg_label = seg_label.reshape(1, seg_label.shape[0], seg_label.shape[1])
            data = {'label': label, 'input': input, 'mask': mask, 'seg_label': seg_label}
        else:
            data = {'label': label, 'input': input, 'mask': mask}

        return data

    def __len__(self):
        return len(self.idx_list)

    def generate_mask(self, input):

        ratio = self.ratio
        size_window = self.size_window
        size_data = self.size_data
        num_sample = int(size_data[2] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[0]):
            idy_msk = np.random.randint(0, size_data[2], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[2] - (idy_msk_neigh >= size_data[2]) * size_data[2]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (ich, idy_msk, idx_msk)
            id_msk_neigh = (ich, idy_msk_neigh, idx_msk_neigh)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask

class Test_Loader(Dataset):
    def __init__(self, data_path, sgm=25, size_data=(1, 512, 512), ratio=0.9, size_window=(5, 5)):

        self.data_path = data_path
        self.sgm = sgm
        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window
        self.idx_list = []

        for root,dir,files in os.walk(self.data_path):
            if files:
                for file in files:
                    self.idx_list.append(os.path.join(root,file))

    def __getitem__(self, index):
        image = Image.open(self.idx_list[index])
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.array(image)
        image = image.reshape(1, image.shape[0], image.shape[1])

        label = image
        # randomly mask
        input, mask = self.generate_mask(copy.deepcopy(label))
        # for prediction only
        input2, _ = self.generate_mask(copy.deepcopy(label))
        input3, _ = self.generate_mask(copy.deepcopy(label))
        input4, _ = self.generate_mask(copy.deepcopy(label))

        data = {'input4':input4, 'input3':input3, 'input2':input2, 'name': self.idx_list[index], 'image': image, 'label': label, 'input': input, 'mask': mask}

        return data

    def __len__(self):

        return len(self.idx_list)

    def get_name(self):
        return self.idx_list

    def generate_mask(self, input):

        ratio = self.ratio
        size_window = self.size_window
        size_data = self.size_data
        num_sample = int(size_data[2] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[0]):
            idy_msk = np.random.randint(0, size_data[2], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[2] - (idy_msk_neigh >= size_data[2]) * size_data[2]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (ich, idy_msk, idx_msk)
            id_msk_neigh = (ich, idy_msk_neigh, idx_msk_neigh)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask

class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = self.std * data + self.mean
        return data
