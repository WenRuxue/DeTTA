# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import SimpleITK as itk
import numpy as np
import cv2
# slice of the .nii to .png

def slice_nii(filepath, savepath, name_length,mask=False):
    filename = os.listdir(filepath)
    for i in filename:
        path = os.path.join(filepath, i)
        img = itk.ReadImage(path)
        array = itk.GetArrayFromImage(img)
        i = i[:-4]
        num = int(i[name_length:])
        for n_slice in range(array.shape[0]):
            result_path = savepath + "%03d-%03d" %(num,n_slice) + ".png"
            print(result_path)
            if mask:
                mask = array[n_slice]
                mask[mask >= 1] = 1
                cv2.imwrite(result_path, mask)
            else:
                image = array[n_slice]
                image[image > 400] = 400
                image[image<-200]=-200
                image = (image - np.min(image))/(np.max(image) - np.min(image))*255
                cv2.imwrite(result_path, image)

filepath_unlabel = '../../LITSChallenge/Test-Data'
savepath_unlabel = '../../LITSChallenge/Dataset/unlabel_data/image/'
filepath_img = '../../LITSChallenge/Training-Batch/img'
savepath_img = '../../LITSChallenge/Dataset/label_data/image/'
filepath_seg = '../../LITSChallenge/nii/Training-Batch/seg'
savepath_seg = '../../LITSChallenge/Dataset/label_data/label/'

slice_nii(filepath_seg,savepath_seg,13,mask=True) # name_length is set according to the nii name. Finally 001_001.png, representing the 001 slice of the 001 volume
slice_nii(filepath_img,savepath_img,13)
slice_nii(filepath_unlabel,savepath_unlabel,13)
