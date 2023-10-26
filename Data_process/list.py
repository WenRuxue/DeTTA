# encoding: utf-8
import os
import random
from shutil import copy
import numpy as np

def to_text(data, root_dir, num_list):
    txt,val_txt = [],[]
    num_val_list = random.sample(num_list,2)
    for root,dir,files in os.walk(data):
        if files:
            for file in files:
                if file.endswith('.png'):
                    if int(file[:3]) in num_list:
                        if int(file[:3]) in num_val_list:
                            val_txt.append(file)
                        txt.append(file)
    print(num_list,txt,len(txt))
    print(num_val_list,val_txt,len(val_txt))
    fo1 = open(root_dir + '/train_full_list.txt', 'w')
    fo2 = open(root_dir + '/val_full_list.txt', 'w')

    for item in txt:
        if item in val_txt:
            fo2.write(str(item) + '\n')
        else:
            fo1.write(str(item) + '\n')

def unlabel_to_text(root_dir, unlabel_data_path):
    unlabel_txt = []
    unlabel_filenames = os.listdir(unlabel_data_path)

    for item in unlabel_filenames:
        if item.endswith('.png'):
            unlabel_txt.append(item)

    fo1 = open(root_dir + '/unlabel_list.txt', 'w')
    for item in unlabel_txt:
        fo1.write(str(item) + '\n')


if __name__ == "__main__":

    data = r'/Data/data/CHAOS_png/Dataset/label_data/image/'
    root_dir = data.replace('image/','')
    num_list = [1,5,6,8,14,16,18,19,21,22,23,24,26,27,29,30]
    num_new_list = random.sample(num_list, 12) # Select train or val nums, 12 for training, 4 for validating
    to_text(data, root_dir, num_new_list)

    unlabel_data_path = r'/Data/data/CHAOS_png/Dataset/unlabel_data/image/'
    root_dir = data.replace('image/', '')
    unlabel_to_text(root_dir, unlabel_data_path)